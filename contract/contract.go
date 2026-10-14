// Package contract holds the types that cross the boundary between this ALPR service and the
// application that embeds it.
//
// The service reads streams, detects plates and reads them. That is all it does. It does not
// decide billing, organisation, user, alert, authorisation, persistence, commercial webhook,
// plan or subscription -- those belong to the application, and every one of them is a reason
// this package must not grow a database handle, an HTTP client or a tenant id.
//
// Nothing here imports anything outside the standard library, and nothing here imports another
// package of this service. Everything else signs against these types, so a change here is a
// change to the whole service.
package contract

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"time"
)

// BBox is a rectangle in the coordinates of the original frame.
//
// Detections never leave this service in letterbox coordinates. The letterbox is an
// implementation detail of the detector's input tensor, and a box expressed in it is wrong in a
// way that looks right: the numbers are plausible, they just point somewhere else.
type BBox struct {
	X, Y, Width, Height float64
}

func (b BBox) X2() float64 { return b.X + b.Width }
func (b BBox) Y2() float64 { return b.Y + b.Height }

// CameraSpec is one camera the application wants processed.
//
// The ID is opaque to this service: it is carried through to results and metrics and never
// parsed. Revision is what makes reconciliation safe -- see Source.
type CameraSpec struct {
	ID string

	// Revision changes when anything about this camera changed. The supervisor restarts a
	// camera when its revision moves and leaves it alone when it does not.
	//
	// Without it, reconciliation has to compare every field to decide whether a restart is
	// needed, and any field the comparison forgets becomes a camera that silently keeps
	// running with stale settings.
	Revision uint64

	URL      string
	Username string
	Password string

	Enabled bool

	// FrameRate is the rate the camera is expected to deliver. It is a hint for diagnostics,
	// not a throttle: this service takes the newest frame available and never paces itself.
	FrameRate float64

	// MinDetectInterval is how long to wait between two detection passes on this camera.
	//
	// The Python service used a frame counter (`frame_count % FRAME_SKIP`). That only means
	// something when the consumer sees every frame; under latest-frame-wins the counter has
	// let go of the clock, and the same setting behaves differently on a 2fps camera and a
	// 30fps one. Time is the honest unit.
	MinDetectInterval time.Duration

	// Brightness and Contrast are percentages; 100 means untouched, and untouched costs
	// nothing because the adjustment is skipped entirely.
	Brightness int
	Contrast   int

	// Placement optionally pins this camera to a device ("cuda:1"). Empty lets the service
	// choose the least loaded one.
	Placement string
}

// redacted replaces a credential wherever a camera is printed.
const redacted = "[redacted]"

// redactedIfSet keeps "absent" readable while hiding the value.
func redactedIfSet(s string) string {
	if s == "" {
		return ""
	}
	return redacted
}

// LogValue prints the camera, not its credentials.
//
// The URL is left out with the password: a stream URL routinely carries the credentials in its own
// userinfo, and a log line about a camera needs its id, not its address.
func (s CameraSpec) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("id", s.ID),
		slog.Uint64("revision", s.Revision),
		slog.Bool("enabled", s.Enabled),
		slog.String("placement", s.Placement),
	)
}

// MarshalJSON keeps the credentials out of anything that serialises a camera.
func (s CameraSpec) MarshalJSON() ([]byte, error) {
	// A local type with no methods: converting to it is what stops MarshalJSON calling itself.
	type plain CameraSpec
	safe := plain(s)
	safe.URL = redactedIfSet(s.URL)
	safe.Username = redactedIfSet(s.Username)
	safe.Password = redactedIfSet(s.Password)
	return json.Marshal(safe)
}

// Source is how the application tells the service which cameras to run.
//
// Cameras returns the entire desired set, every time. It is a snapshot, never a delta.
//
// A delta interface would need the application to guarantee that no change is ever missed, and
// a missed delta is a camera that stays up after being deleted or stays down after being fixed
// -- with nothing in either system showing a disagreement. A snapshot cannot drift: whatever
// the application returns is what runs after the next reconcile.
type Source interface {
	Cameras(ctx context.Context) ([]CameraSpec, error)
}

// Watcher is optional and shortens the wait between reconciles.
//
// Changed signals that the set *may* have changed. It is a hint and never the truth -- the
// service still calls Cameras on its own schedule, and a Watcher that never fires only makes
// changes take longer to apply, never lost.
//
// This mirrors how the Python service worked: a 60s sync loop plus Postgres LISTEN/NOTIFY with
// a one-second debounce against notify storms. The debounce lives in this service, so an
// application implementing Changed may fire as often as it likes.
type Watcher interface {
	Changed(ctx context.Context) <-chan struct{}
}

// ModelInfo identifies what produced a reading, so a result can be attributed to an artefact.
type ModelInfo struct {
	Detector string // sha256 prefix of the detector .ort
	OCR      string // sha256 prefix of the ocr .ort
	Device   int    // which GPU ran it
}

// Detection is one accepted plate reading, complete and owned by the receiver.
type Detection struct {
	CameraID string

	Plate string
	// CharConfidences holds one probability per character of Plate, in reading order.
	CharConfidences []float64
	// Confidence is the mean of CharConfidences.
	//
	// Mean, not minimum: one weak character in an otherwise certain plate is usually a smudge
	// on one digit, and taking the minimum throws away the whole reading over it.
	Confidence float64

	// Region is the country the OCR's region head chose, and RegionConfidence its probability.
	//
	// There is no plate-format validation anywhere in this service. This product reads plates
	// worldwide; a national format check would discard every foreign plate the model read
	// correctly, silently and without a log.
	Region           string
	RegionConfidence float64

	Box BBox
	// DetectorScore is the detector's confidence in the box, unrelated to Confidence above.
	DetectorScore float64

	// CapturedAt is when the frame left the camera; DetectedAt is when this reading was
	// produced. The difference is the end-to-end age, which is what an operator feels.
	CapturedAt time.Time
	DetectedAt time.Time

	Model ModelInfo

	// FullJPEG and CropJPEG are already encoded, and empty unless the application asked for
	// them. They are plain values: the receiver owns them and may keep them forever.
	//
	// Handing over a borrowed frame instead would be cheaper and would come with a lifetime
	// rule ("invalid after this call returns"). A lifetime rule that crosses a module boundary
	// is a rule somebody breaks, and breaking this one corrupts an image rather than failing.
	FullJPEG []byte
	CropJPEG []byte
}

// Sink is how accepted readings leave the service.
//
// Detect is called ON the camera's own goroutine and blocks that camera until it returns.
// Return quickly or return an error.
//
// The service deliberately does none of the work that follows a detection: it does not upload,
// does not notify, does not call anything, and starts no goroutine that outlives this call. It
// hands over a complete Detection and stops.
//
// That boundary exists because of a specific production failure. In the Python service the
// image upload was fired before the database insert, to save the notification a wait. Under
// load the insert took longer than the receiving endpoint's retry budget (5 attempts, 500ms
// apart), the backend answered 404, and the notification was lost outright -- delivery fell
// from 100% to under 15% in half an hour, on 19/08.
//
// Whatever ordering an application needs, it writes inside Detect, with the row and the id in
// hand. From inside this service the wrong order is not expressible.
type Sink interface {
	Detect(ctx context.Context, d Detection) error
}

// ErrBackpressure tells the service the application cannot keep up.
//
// A Sink returning it makes the camera raise its detection interval for a while, which is
// backpressure crossing the boundary explicitly instead of as a growing queue somewhere.
var ErrBackpressure = errors.New("alpr: sink is applying backpressure")

// Event is something worth telling the application about a camera's life.
type Event struct {
	CameraID string
	Kind     EventKind
	// Message is safe to show a human. It never contains the stream URL: that URL carries
	// credentials, and these messages end up in operator-facing logs and panels.
	Message string
	At      time.Time
}

type EventKind string

// There is deliberately no "camera down".
//
// A camera in this service never goes down: it keeps dialling, at an interval that grows to a
// ceiling and stays there, until it answers. The event existed when the loop gave up after a
// handful of failed dials -- and giving up is what then required a watchdog to notice, a status
// column to write, a second query to find the camera again and a ceiling to stop retrying it.
// Removing the giving-up removed all of them.
//
// EventStreamReconnect is what a caller watches instead. It says the same thing an operator
// needs -- this camera is not answering -- without asserting a final state that is not final.
const (
	EventCameraUp        EventKind = "camera_up"
	EventCameraStale     EventKind = "camera_stale"
	EventStreamReconnect EventKind = "stream_reconnect"
	EventInferenceError  EventKind = "inference_error"
)

// Observer is optional and receives camera lifecycle events.
type Observer interface {
	Observe(ctx context.Context, e Event)
}

// Preview is optional and receives frames for a live overlay.
//
// It is best-effort by contract: the service never waits for a preview consumer, and drops
// frames rather than let a slow viewer slow down detection. A preview is something a person is
// watching; a detection is the product.
type Preview interface {
	Frame(cameraID string, jpeg []byte, boxes []BBox)
}
