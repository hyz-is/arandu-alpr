//go:build unix

package measure

import (
	"syscall"
	"time"
)

// selfCPUTime is user+system time consumed by this process so far.
//
// getrusage rather than parsing /proc/self/stat: it is one syscall, it is the same call on
// Linux and on the Mac this harness is written on, and it needs no clock-tick constant that is
// right on one of the two.
func selfCPUTime() (time.Duration, error) {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0, err
	}
	return timevalDuration(usage.Utime) + timevalDuration(usage.Stime), nil
}

func timevalDuration(tv syscall.Timeval) time.Duration {
	return time.Duration(tv.Sec)*time.Second + time.Duration(tv.Usec)*time.Microsecond
}
