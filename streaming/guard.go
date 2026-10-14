package streaming

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/netip"
	"net/url"
	"strings"
)

// ErrCameraAddress is a camera address the engine refuses to connect to.
var ErrCameraAddress = errors.New("streaming: camera address refused")

// protocolWhitelist is every protocol ffmpeg may open for a guarded camera: the four schemes a
// camera is reached by and what they run on underneath (RTP and UDP inside RTSP, TLS under rtsps
// and https, crypto under an encrypted HLS segment). It is what keeps file:, concat:, subfile: and
// the rest out -- including when they are named not in the camera URL but inside what the camera
// sends back, a playlist being the obvious carrier.
const protocolWhitelist = "rtsp,rtsps,rtp,srtp,udp,tcp,tls,http,https,httpproxy,crypto"

// Guard decides which camera addresses the engine connects to.
//
// The application that writes the camera file refuses internal addresses when a camera is saved,
// but it cannot see two things: a name that resolves to a public address when it is saved and to
// an internal one later, and what ffmpeg is told to fetch next. Only the process that connects
// sees the address at the moment it connects, so the check is repeated here, on every open.
//
// AllowPrivate turns it off, for an engine that runs inside the customer's own network, where the
// cameras are on private addresses by definition.
type Guard struct {
	AllowPrivate bool
	// Lookup resolves a host name; nil uses the system resolver.
	Lookup func(ctx context.Context, host string) ([]net.IPAddr, error)
}

// Address returns the URL ffmpeg should open for raw and the input options that go before it.
//
// Only rtsp, rtsps, http and https are accepted, and every address the host resolves to has to be
// public. An RTSP URL with a host name is rewritten to the address that was checked, so the
// connection goes where the check looked and a name that changes its answer between the two
// cannot redirect it. HTTP keeps its name, because a server behind a name may need it in the Host
// header and TLS needs it for the certificate; for those the check narrows the window rather than
// closing it.
func (g Guard) Address(ctx context.Context, raw string) (string, []string, error) {
	if g.AllowPrivate {
		return raw, nil, nil
	}
	u, err := url.Parse(raw)
	if err != nil {
		return "", nil, fmt.Errorf("%w: not a URL", ErrCameraAddress)
	}
	scheme := strings.ToLower(u.Scheme)
	switch scheme {
	case "rtsp", "rtsps", "http", "https":
	default:
		return "", nil, fmt.Errorf("%w: scheme %q is not rtsp, rtsps, http or https", ErrCameraAddress, u.Scheme)
	}
	host := u.Hostname()
	if host == "" {
		return "", nil, fmt.Errorf("%w: no host", ErrCameraAddress)
	}

	var addrs []netip.Addr
	if literal, err := netip.ParseAddr(host); err == nil {
		addrs = []netip.Addr{literal}
	} else {
		lookup := g.Lookup
		if lookup == nil {
			lookup = net.DefaultResolver.LookupIPAddr
		}
		found, err := lookup(ctx, host)
		if err != nil || len(found) == 0 {
			return "", nil, fmt.Errorf("%w: %s does not resolve", ErrCameraAddress, host)
		}
		for _, a := range found {
			if addr, ok := netip.AddrFromSlice(a.IP); ok {
				addrs = append(addrs, addr)
			}
		}
	}
	if len(addrs) == 0 {
		return "", nil, fmt.Errorf("%w: %s has no usable address", ErrCameraAddress, host)
	}
	for _, addr := range addrs {
		if internal(addr) {
			return "", nil, fmt.Errorf("%w: %s is an internal address", ErrCameraAddress, addr)
		}
	}

	if scheme == "rtsp" && host != addrs[0].Unmap().String() {
		pinned := addrs[0].Unmap().String()
		if addrs[0].Unmap().Is6() {
			pinned = "[" + pinned + "]"
		}
		if port := u.Port(); port != "" {
			pinned = net.JoinHostPort(addrs[0].Unmap().String(), port)
		}
		u.Host = pinned
		raw = u.String()
	}
	return raw, []string{"-protocol_whitelist", protocolWhitelist}, nil
}

// internalPrefixes are the ranges a camera on the internet is never in, beyond what netip's own
// predicates already name: shared address space for carrier NAT, the IETF protocol block, the
// benchmarking range, the reserved class E, "this network", and NAT64 -- which embeds an IPv4
// address that has to be judged as itself.
var internalPrefixes = []netip.Prefix{
	netip.MustParsePrefix("0.0.0.0/8"),
	netip.MustParsePrefix("100.64.0.0/10"),
	netip.MustParsePrefix("192.0.0.0/24"),
	netip.MustParsePrefix("198.18.0.0/15"),
	netip.MustParsePrefix("240.0.0.0/4"),
	netip.MustParsePrefix("64:ff9b::/96"),
}

// internal reports whether addr is anywhere a camera on the internet cannot be: loopback,
// private, link-local (which holds the cloud metadata endpoints), multicast, unspecified, or one
// of internalPrefixes. An IPv4 address mapped into IPv6 is judged as the IPv4 address it is.
func internal(addr netip.Addr) bool {
	addr = addr.Unmap()
	if addr.IsLoopback() || addr.IsPrivate() || addr.IsLinkLocalUnicast() || addr.IsLinkLocalMulticast() ||
		addr.IsInterfaceLocalMulticast() || addr.IsMulticast() || addr.IsUnspecified() {
		return true
	}
	for _, p := range internalPrefixes {
		if p.Contains(addr) {
			return true
		}
	}
	return false
}
