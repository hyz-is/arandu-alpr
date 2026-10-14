//go:build cuda

package main

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// publisher is one ffmpeg process replaying a clip as an RTSP stream.
type publisher struct {
	url  string
	cmd  *exec.Cmd
	logs *strings.Builder
}

// publisherSet is every replay for one sweep step.
type publisherSet struct {
	pubs   []*publisher
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// startPublishers replays clips as RTSP feeds, one process per camera.
//
// # Why -re is not optional
//
// ffmpeg reads a local file as fast as the disk allows. Without `-re` a ten-second clip is
// pushed in a fraction of a second and then the input ends, so the fleet receives one burst
// followed by silence -- and every rate computed from that is a rate of nothing. `-re` paces the
// input at its native frame rate, which is what a camera does, and `-stream_loop -1` keeps it
// going, which is also what a camera does. Both are recorded in the report so a line produced
// without them can be thrown away rather than believed.
//
// # Why an external RTSP server is required
//
// ffmpeg can listen for one client itself (`-rtsp_flags listen`), and that is not enough here.
// Opening a camera is TWO connections: ffprobe asks for the geometry and disconnects, then
// ffmpeg connects to read frames. A single-client listener serves the probe and refuses the
// capture, so every camera fails to open for a reason that looks like a broken URL. A real RTSP
// server -- mediamtx is one line to start -- accepts both.
//
// # Why one process per camera rather than one clip fanned out
//
// A fleet's cost includes decoding N independent streams. Publishing one path and pointing
// forty cameras at it would measure a fleet that shares a decoder, which is not the fleet
// anybody runs.
func startPublishers(ctx context.Context, cfg sweepConfig, cameras int) (*publisherSet, error) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return nil, fmt.Errorf("fleet: ffmpeg is not on PATH: %w", err)
	}
	if err := reachable(cfg.RTSPServer, 5*time.Second); err != nil {
		return nil, err
	}

	runCtx, cancel := context.WithCancel(ctx)
	set := &publisherSet{cancel: cancel}

	for i := 0; i < cameras; i++ {
		clip := cfg.Clips[i%len(cfg.Clips)]
		if _, err := os.Stat(clip); err != nil {
			cancel()
			return nil, fmt.Errorf("fleet: clip %s: %w", clip, err)
		}

		target := fmt.Sprintf("%s/bench%03d", strings.TrimRight(cfg.RTSPServer, "/"), i)
		args := []string{
			"-hide_banner", "-loglevel", "error",
			"-re",                // pace the file at its native rate; see above
			"-stream_loop", "-1", // and never let it end
			"-i", clip,
			"-an",
			// Copy rather than re-encode. Re-encoding forty clips would put the benchmark's own
			// CPU cost inside the measurement, and the fleet's CPU budget is one of the numbers
			// being reported.
			"-c:v", "copy",
			"-f", "rtsp",
			"-rtsp_transport", "tcp",
			target,
		}

		cmd := exec.CommandContext(runCtx, "ffmpeg", args...)
		logs := &strings.Builder{}
		cmd.Stderr = logs
		if err := cmd.Start(); err != nil {
			cancel()
			set.stop()
			return nil, fmt.Errorf("fleet: starting publisher %d: %w", i, err)
		}

		pub := &publisher{url: target, cmd: cmd, logs: logs}
		set.pubs = append(set.pubs, pub)
		set.wg.Add(1)
		go func() {
			defer set.wg.Done()
			_ = cmd.Wait()
		}()
	}

	// A publisher that dies immediately -- a codec the server refuses, a path already taken --
	// must be reported here rather than as forty cameras that cannot dial. The second symptom
	// points at the fleet; only the first points at the cause.
	time.Sleep(2 * time.Second)
	for i, pub := range set.pubs {
		if pub.cmd.ProcessState != nil && pub.cmd.ProcessState.Exited() {
			set.stop()
			return nil, fmt.Errorf("fleet: publisher %d exited immediately (%s):\n%s",
				i, pub.url, strings.TrimSpace(pub.logs.String()))
		}
	}
	return set, nil
}

// urls lists what the cameras should be pointed at.
func (s *publisherSet) urls() []string {
	out := make([]string, 0, len(s.pubs))
	for _, p := range s.pubs {
		out = append(out, p.url)
	}
	return out
}

// stop kills every publisher and waits for it.
//
// Waiting matters: a sweep runs step after step and an ffmpeg still holding an RTSP path makes
// the next step's publisher fail with a message about the path rather than about the leak.
func (s *publisherSet) stop() {
	s.cancel()
	s.wg.Wait()
}

// reachable checks that something is listening where the clips are to be published.
//
// Checked before anything starts, because the alternative failure is N cameras timing out on
// dial one after another -- which takes minutes and reads as a broken fleet rather than as a
// missing server.
func reachable(rtsp string, timeout time.Duration) error {
	parsed, err := url.Parse(rtsp)
	if err != nil {
		return fmt.Errorf("fleet: -rtsp-server %q is not a URL: %w", rtsp, err)
	}
	host := parsed.Host
	if parsed.Port() == "" {
		host = net.JoinHostPort(parsed.Hostname(), "8554")
	}

	conn, err := net.DialTimeout("tcp", host, timeout)
	if err != nil {
		return fmt.Errorf("fleet: nothing is listening at %s: %w\n"+
			"    The sweep publishes clips to an RTSP server and reads them back as cameras.\n"+
			"    Start one first, for example:  mediamtx", host, err)
	}
	return conn.Close()
}
