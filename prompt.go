package mneme

import (
	"fmt"
	"strings"

	"github.com/AccursedGalaxy/mneme/types"
)

// extractionPromptV1 is mneme's own extraction system prompt. It is versioned:
// the eval harness scores each version so prompt changes are decisions backed
// by numbers, not vibes. This is the core of "owning the IP" — it reimplements
// the ideas of a good additive-extraction prompt in our own words, not a copy
// of any vendor's text.
const extractionPromptV1 = `You extract durable facts worth remembering from a conversation.

Your job: read the NEW MESSAGES and output the long-term facts they establish —
the things a future assistant would need to recall to serve this person well.

WHAT TO EXTRACT
- Stable personal or world facts: identity, location, job, possessions, health.
- Preferences, likes and dislikes, opinions held.
- Plans, goals, intentions, commitments, and decisions made.
- Relationships between people, and named entities (people, products, places,
  organizations, projects).
- Specific quantities, titles, dates, and numbers.

WHAT TO SKIP
- Greetings, small talk, filler, acknowledgements, and transient pleasantries.
- Questions the user asks (a question is not a fact unless it reveals one).
- Anything already covered by the EXISTING MEMORIES shown below. Do not re-emit a
  fact that is already known — that is the single most important rule for keeping
  memory clean.

HOW TO WRITE EACH FACT
- Self-contained: it must stand alone with no outside context. Resolve every
  pronoun to a name or role. Never write "he", "she", "it", "they", or "this" —
  a stranger reading the fact cold must understand it fully.
- Specific, not vague: keep proper nouns, exact quantities, titles, and
  qualifiers. Write "assistant manager", not "manager"; "Ferrari 488 GTB", not
  "a car"; "416 pages", not "about 400". Specificity is what makes a fact
  retrievable later.
- Meaning-preserving: capture the exact claim. Do not invert, soften, or
  exaggerate it. "used to love sushi but stopped" is not "loves sushi".
- One fact per item: a message about several topics yields several facts. Cover
  every topic in the message — do not stop after the first one.
- No fabrication: every fact must trace to something actually said. Do not infer
  attributes (gender, age, nationality, etc.) that were not stated.
- Date grounding: convert relative dates ("yesterday", "last week", "next
  month") to absolute calendar dates using the OBSERVATION DATE provided.

OUTPUT
Return ONLY a JSON object, no prose and no code fences, in exactly this shape:
{"memory":[{"id":"0","text":"<a self-contained fact>","attributed_to":"user"}]}
- "id" is a sequential string starting at "0".
- "attributed_to" is "user" or "assistant" — who the fact is about / who said it.
- If there is nothing worth remembering, return {"memory":[]}.`

// DefaultPromptVersion is the prompt version New uses unless WithPromptVersion
// overrides it.
const DefaultPromptVersion = "v1"

// promptVersions maps a version name to its extraction system prompt. The eval
// harness iterates this so each version gets a recorded score; a prompt change
// is a new entry, never an in-place edit of a shipped one.
var promptVersions = map[string]string{
	"v1": extractionPromptV1,
}

// PromptVersions returns the registered prompt version names (unordered).
func PromptVersions() []string {
	out := make([]string, 0, len(promptVersions))
	for k := range promptVersions {
		out = append(out, k)
	}
	return out
}

// systemPrompt returns the extraction system prompt for a version, falling back
// to the default when the version is unknown.
func systemPrompt(version string) string {
	if p, ok := promptVersions[version]; ok {
		return p
	}
	return promptVersions[DefaultPromptVersion]
}

// consolidationPromptV1 is mneme's own consolidation system prompt, used under
// WithStrategy(Consolidate). It reimplements the idea of mem0's ADD/UPDATE/
// DELETE/NONE reconciliation in our own words (orientation only — see PLAN.md
// §5). It is versioned exactly like the extraction prompt so a change is a
// measured decision.
//
// It lives in its own registry (consolidationPrompts), NOT in promptVersions:
// promptVersions is iterated by the eval harness as *extraction* system prompts,
// so mixing a consolidation prompt into it would make eval score this prompt as
// an extractor. Same versioning mechanism, separate namespace.
const consolidationPromptV1 = `You maintain a person's long-term memory.

You are given the CURRENT MEMORIES (each with a numeric id) and a set of NEW
FACTS just extracted from a conversation. Decide how the new facts change the
stored memory, and output one operation per change.

EVENTS (choose exactly one per operation):
- ADD: the new fact is genuinely new information not covered by any current
  memory. Use id "new" and put the new fact in "text".
- UPDATE: the new fact changes the value of an existing memory — a move, a new
  job, a changed preference, a corrected or newly precise detail. Reference the
  existing memory's id and write the full corrected fact in "text". Prefer
  UPDATE over deleting and re-adding when the same underlying fact changed.
- DELETE: an existing memory is explicitly contradicted or made obsolete by the
  new facts and must be removed. Reference its id.
- NONE: an existing memory is unaffected, or a new fact merely restates
  something already known. This makes no change.

RULES
- Only emit operations that change something. A current memory the new facts do
  not touch should simply be left alone (you may omit it, or mark it NONE).
- Never invent ids. UPDATE, DELETE and NONE must reference an id that appears in
  CURRENT MEMORIES exactly. ADD must use the literal id "new".
- DELETE only on real contradiction or obsolescence — never just because two
  facts are about a similar topic.
- Each resulting fact must stay self-contained and specific: resolve pronouns to
  names, keep proper nouns, quantities and dates — the same standard as
  extraction.

OUTPUT
Return ONLY a JSON object, no prose and no code fences, in exactly this shape:
{"memory":[{"id":"<int|new>","text":"<the resulting fact>","event":"ADD|UPDATE|DELETE|NONE"}]}
If nothing should change, return {"memory":[]}.`

// DefaultConsolidationVersion is the consolidation prompt version used unless
// WithConsolidationVersion overrides it.
const DefaultConsolidationVersion = "v1"

// consolidationPrompts maps a version name to its consolidation system prompt.
// Separate from promptVersions on purpose (see consolidationPromptV1).
var consolidationPrompts = map[string]string{
	"v1": consolidationPromptV1,
}

// ConsolidationPromptVersions returns the registered consolidation prompt
// version names (unordered).
func ConsolidationPromptVersions() []string {
	out := make([]string, 0, len(consolidationPrompts))
	for k := range consolidationPrompts {
		out = append(out, k)
	}
	return out
}

// consolidationSystemPrompt returns the consolidation system prompt for a
// version, falling back to the default when the version is unknown or empty.
func consolidationSystemPrompt(version string) string {
	if p, ok := consolidationPrompts[version]; ok {
		return p
	}
	return consolidationPrompts[DefaultConsolidationVersion]
}

// buildConsolidationUser assembles the user-message half of the consolidation
// call: the integer-labelled current memories and the newly extracted facts.
// The durable instructions live in consolidationPromptV1 (the system message).
func buildConsolidationUser(current []labeledMemory, candidates []extractedFact) string {
	var b strings.Builder

	b.WriteString("CURRENT MEMORIES (id: fact):\n")
	if len(current) == 0 {
		b.WriteString("(none)\n")
	} else {
		for _, m := range current {
			fmt.Fprintf(&b, "%s: %s\n", m.ID, m.Text)
		}
	}
	b.WriteString("\n")

	b.WriteString("NEW FACTS (just extracted from the conversation):\n")
	if len(candidates) == 0 {
		b.WriteString("(none)\n")
	} else {
		for _, c := range candidates {
			fmt.Fprintf(&b, "- %s\n", c.Text)
		}
	}

	return b.String()
}

// labeledMemory is an existing fact shown to the extractor under a small integer
// id. The integer relabel (real UUID -> "0","1",...) is a load-bearing
// anti-hallucination trick: LLMs copy long UUIDs unreliably, so we never put
// real ids in the prompt.
type labeledMemory struct {
	ID   string
	Text string
}

// buildExtractionUser assembles the user-message half of the extraction call:
// the observation date, the integer-labelled existing memories, recent context
// for reference resolution, and the new messages to extract from. The durable
// instructions live in extractionPromptV1 (the system message); this is only the
// per-call data.
func buildExtractionUser(today string, existing []labeledMemory, context, conv []types.Message) string {
	var b strings.Builder

	fmt.Fprintf(&b, "OBSERVATION DATE: %s\n\n", today)

	b.WriteString("EXISTING MEMORIES (already known — do not re-emit these):\n")
	if len(existing) == 0 {
		b.WriteString("(none)\n")
	} else {
		for _, m := range existing {
			fmt.Fprintf(&b, "%s: %s\n", m.ID, m.Text)
		}
	}
	b.WriteString("\n")

	if ctx := renderMessages(context); ctx != "" {
		b.WriteString("RECENT CONTEXT (for reference only — do not extract from this):\n")
		b.WriteString(ctx)
		b.WriteString("\n\n")
	}

	b.WriteString("NEW MESSAGES (extract durable facts from these):\n")
	if conv := renderMessages(conv); conv != "" {
		b.WriteString(conv)
	} else {
		b.WriteString("(none)")
	}
	b.WriteString("\n")

	return b.String()
}

// renderMessages formats messages as "role (name): content" lines, skipping
// system messages (ignored for extraction) and empty content.
func renderMessages(msgs []types.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		if m.Role == "system" || strings.TrimSpace(m.Content) == "" {
			continue
		}
		role := m.Role
		if role == "" {
			role = "user"
		}
		if m.Name != "" {
			fmt.Fprintf(&b, "%s (%s): %s\n", role, m.Name, m.Content)
		} else {
			fmt.Fprintf(&b, "%s: %s\n", role, m.Content)
		}
	}
	return strings.TrimRight(b.String(), "\n")
}
