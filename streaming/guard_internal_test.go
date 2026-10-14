package streaming

import (
	"context"
	"errors"
	"net"
	"net/netip"
	"strings"
	"testing"
)

// resolver answers from a table, the way a DNS server under someone else's control would.
func resolver(answers map[string]string) func(context.Context, string) ([]net.IPAddr, error) {
	return func(_ context.Context, host string) ([]net.IPAddr, error) {
		a, ok := answers[host]
		if !ok {
			return nil, errors.New("no such host")
		}
		var out []net.IPAddr
		for _, ip := range strings.Split(a, ",") {
			out = append(out, net.IPAddr{IP: net.ParseIP(ip)})
		}
		return out, nil
	}
}

func TestTheGuardRefusesWhatACameraOnTheInternetCannotBe(t *testing.T) {
	g := Guard{Lookup: resolver(map[string]string{
		"rebound.example.com":  "203.0.113.10,10.0.0.5",
		"metadata.example.com": "169.254.169.254",
	})}
	for _, raw := range []string{
		"rtsp://127.0.0.1:554/live",
		"rtsp://10.1.2.3/live",
		"rtsp://192.168.0.10/live",
		"rtsp://172.16.5.5/live",
		"rtsp://100.64.1.1/live",        // carrier NAT
		"http://169.254.169.254/latest", // cloud metadata
		"rtsp://[::1]/live",
		"rtsp://[fd00::1]/live",
		"rtsp://[::ffff:10.0.0.1]/live", // IPv4 hiding in IPv6
		"rtsp://0.0.0.0/live",
		"rtsp://metadata.example.com/live",
		"rtsp://rebound.example.com/live", // one internal answer is enough
		"rtsp://nowhere.example.com/live",
		"file:///etc/passwd",
		"concat:rtsp://203.0.113.10/a|file:///etc/passwd",
		"udp://203.0.113.10:1234",
		"rtsp:///live",
	} {
		if _, _, err := g.Address(context.Background(), raw); !errors.Is(err, ErrCameraAddress) {
			t.Errorf("%s was let through: %v", raw, err)
		}
	}
}

func TestTheGuardPinsAnRTSPNameToTheAddressItChecked(t *testing.T) {
	g := Guard{Lookup: resolver(map[string]string{"cam.example.com": "203.0.113.10", "cam6.example.com": "2001:db8::10"})}
	// 2001:db8::/32 is documentation space, which netip does not call private; it stands in here
	// for any public IPv6 address.
	for raw, want := range map[string]string{
		"rtsp://admin:s3cr%40t@cam.example.com:554/Streaming/101": "rtsp://admin:s3cr%40t@203.0.113.10:554/Streaming/101",
		"rtsp://cam.example.com/live":                             "rtsp://203.0.113.10/live",
		"rtsp://cam6.example.com:8554/live":                       "rtsp://[2001:db8::10]:8554/live",
		"rtsp://203.0.113.10/live":                                "rtsp://203.0.113.10/live",
		"https://cam.example.com/mjpeg":                           "https://cam.example.com/mjpeg",
	} {
		got, args, err := g.Address(context.Background(), raw)
		if err != nil {
			t.Errorf("%s: %v", raw, err)
			continue
		}
		if got != want {
			t.Errorf("%s became %s, want %s", raw, got, want)
		}
		if len(args) != 2 || args[0] != "-protocol_whitelist" || strings.Contains(args[1], "file") || strings.Contains(args[1], "concat") {
			t.Errorf("%s: input options %v, want a protocol whitelist without file or concat", raw, args)
		}
	}
}

func TestAnEngineInsideTheCustomersNetworkConnectsAnywhere(t *testing.T) {
	g := Guard{AllowPrivate: true}
	got, args, err := g.Address(context.Background(), "rtsp://192.168.0.10/live")
	if err != nil || got != "rtsp://192.168.0.10/live" || args != nil {
		t.Fatalf("got %q %v %v", got, args, err)
	}
}

func TestTheRefusalNeverCarriesTheCredentials(t *testing.T) {
	g := Guard{Lookup: resolver(map[string]string{"cam.internal": "10.0.0.9"})}
	_, _, err := g.Address(context.Background(), "rtsp://admin:sup3rs3cret@cam.internal/live")
	if err == nil || strings.Contains(err.Error(), "sup3rs3cret") || strings.Contains(err.Error(), "admin") {
		t.Fatalf("the refusal: %v", err)
	}
}

func TestEveryRangeTheGuardNamesIsInternal(t *testing.T) {
	for _, s := range []string{"0.1.2.3", "100.127.255.254", "192.0.0.8", "198.19.0.1", "250.0.0.1", "64:ff9b::a00:1", "224.0.0.1", "ff02::1", "fe80::1"} {
		if !internal(netip.MustParseAddr(s)) {
			t.Errorf("%s is not judged internal", s)
		}
	}
	for _, s := range []string{"203.0.113.10", "8.8.8.8", "2001:4860:4860::8888", "177.10.20.30"} {
		if internal(netip.MustParseAddr(s)) {
			t.Errorf("%s, a public address, is judged internal", s)
		}
	}
}
