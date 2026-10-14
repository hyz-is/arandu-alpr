package alpr

// regions maps the OCR's `region` head to a country name.
//
// **Positional, and the order is load-bearing.** The head emits `[1, 66]` logits and the class is
// the argmax index; this slice is the only thing that turns that index into a name. It is copied
// verbatim from `latin_plate_config_v2.yaml::plate_regions` in fast-plate-ocr -- 65 countries and
// `Unknown` -- and reordering it silently relabels every reading the engine has ever produced.
//
// A test pins the length and several known positions for that reason: a diff that looks like
// tidying (sorting the list) would be undetectable at runtime.
var regions = [66]string{
	"Albania",
	"Andorra",
	"Argentina",
	"Armenia",
	"Australia",
	"Austria",
	"Azerbaijan",
	"Bahrain",
	"Belarus",
	"Belgium",
	"Bosnia and Herzegovina",
	"Brazil",
	"Bulgaria",
	"Cambodia",
	"Canada",
	"Croatia",
	"Cyprus",
	"Czech Republic",
	"Denmark",
	"Estonia",
	"Finland",
	"France",
	"Georgia",
	"Germany",
	"Gibraltar",
	"Greece",
	"Guernsey",
	"Hungary",
	"Iceland",
	"Indonesia",
	"Ireland",
	"Israel",
	"Italy",
	"Latvia",
	"Liechtenstein",
	"Lithuania",
	"Luxembourg",
	"Malaysia",
	"Malta",
	"Mexico",
	"Moldova",
	"Monaco",
	"Montenegro",
	"Netherlands",
	"New Zealand",
	"North Macedonia",
	"Norway",
	"Poland",
	"Portugal",
	"Qatar",
	"Romania",
	"San Marino",
	"Serbia",
	"Singapore",
	"Slovakia",
	"Slovenia",
	"Spain",
	"Sweden",
	"Switzerland",
	"Thailand",
	"Turkey",
	"United States",
	"Ukraine",
	"United Kingdom",
	"Vietnam",
	"Unknown",
}

// RegionCount is the size the model's `region` output must have.
const RegionCount = 66

// RegionName resolves a class index from the `region` head.
//
// An out-of-range index returns "Unknown" rather than panicking: it means the loaded model has a
// different head than this table describes, and a worker should degrade to "country not known"
// instead of dying mid-frame. The mismatch is caught at startup by CheckRegionOutput.
func RegionName(class int) string {
	if class < 0 || class >= len(regions) {
		return "Unknown"
	}
	return regions[class]
}

// CheckRegionOutput reports whether a model's `region` output matches this table.
//
// Called once when a session is built. Without it, a model with a different region head would
// produce confident, wrong country names for every plate -- and nothing in the output would look
// out of place.
func CheckRegionOutput(size int) error {
	if size != RegionCount {
		return errRegionSize(size)
	}
	return nil
}

type regionSizeError int

func (e regionSizeError) Error() string {
	return "alpr: model region head has " + itoa(int(e)) + " classes, expected " + itoa(RegionCount) +
		"; the label table would mislabel every reading"
}

func errRegionSize(size int) error { return regionSizeError(size) }

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
