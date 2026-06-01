package sqlite

import "time"

// parseTime accepts the layout we write (RFC3339Nano) and falls back to plain
// RFC3339 for rows that may have been written without sub-second precision.
func parseTime(s string) (time.Time, error) {
	if t, err := time.Parse(timeLayout, s); err == nil {
		return t, nil
	}
	return time.Parse(time.RFC3339, s)
}
