//go:build !unix

package measure

import (
	"errors"
	"time"
)

// selfCPUTime has no portable answer off unix.
//
// It returns an error rather than zero so the sampler records a note. A CPU figure of 0% in a
// report is read as "the box was idle", which is a different claim from "nobody measured".
func selfCPUTime() (time.Duration, error) {
	return 0, errors.New("measure: process CPU time is not available on this platform")
}
