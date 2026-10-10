# CML abstract prompt — `abstract_v1.5`

Changes from v1.4: v1.4 writers (aim `target_center` − 10) landed 12 under to 5 over the center, 77% of them
under it; the contract wants roughly half under. v1.5 moves the aim to `target_center` − 6. Nothing else changes.

Changes from v1.3 to v1.4: in the v1.3 re-pilot every abstract (30/30, three writers) came out 22-39 words above
`target_center`; told that the center was "not a minimum", writers aimed at the top of the range. v1.4 anchors the
ideal length below the center. Everything else is unchanged from v1.3.

Changes from v1.2 to v1.3 (independent evaluator's pilot audit): every v1.2 abstract sat above its length center
and none was a single paragraph, while 89% of the original dataset's summaries are one paragraph; the
audit's minor errors were adjacent beats swapped, actions attributed to the wrong character, implied
outcomes stated as fact, and visions retold as speech; the most common coverage gap was a dropped final scene.

You write the `summary` (abstract) field of one CML-Dataset item.

**Input:** the movie title and a contiguous excerpt of 12–24 scenes from a human-written screenplay, in
Cinematic Markup Language: `<scene>`, `<stage_direction>` (scene heading), `<scene_description>`,
`<character>`, `<parenthetical>`, `<dialogue>`.

**Purpose:** the abstract is the *only* input a screenwriting model gets when CML-Bench asks it to
regenerate this excerpt. It must capture what happens, who is involved, and in what order.

## Requirements

1. **Length.** The manifest gives `target_center` (~145 words for a typical excerpt) and `target_words`
   (center ±35; hard limits 90–300). **Aim for `target_center` − 6.** Anything from `target_center` − 14 to
   `target_center` + 2 is ideal, and about half of your abstracts should end up shorter than `target_center`.
   Go above `target_center` + 10 only for an unusually eventful excerpt, never above `target_words`. Reach the
   length by choosing which beats to state and how tightly, not by padding: cut adjectives, mood and commentary
   first. Count words before saving and trim if you are over.
2. **Shape.** Write **one paragraph** of plain prose by default. Use two paragraphs only if the abstract
   is longer than 200 words and the excerpt has a clear break in place or time; never more than two.
   No title, bullets, headings, markdown, or surrounding quotes.
3. **Coverage.** Follow the excerpt from its first scene to its last. Cover the opening situation, each
   significant turn, conflict, decision or revelation, and where things stand at the end. **The last two
   scenes must each be reflected** — check them before saving. Do not stop halfway.
4. **Order.** Narrate beats in the order the scenes appear. Use "Meanwhile" or "Earlier" only when the
   scene headings or descriptions say the action is simultaneous or earlier; otherwise use plain
   sequence ("Later,", "Back at the motel,", "That night,").
5. **Faithfulness.** Use only what the excerpt shows or states. Do not add events, motives, backstory,
   relationships, names or outcomes from your own knowledge of the film, and do not foreshadow anything
   after the excerpt. Never mention actors, crew, the film's reception, or the film's title. Real people
   who appear as characters in the excerpt are characters and may be named. In particular:
   - **Implied is not shown.** If a death, injury, crime or outcome is only implied (a scream off-screen,
     a gun raised, a cut away), describe what is shown ("raises the gun as the scene cuts away"), not the
     inferred result ("kills him").
   - **Who did it.** Before saving, check every "X does/offers/demands/suggests Y" against the scene:
     the character who initiates an action, offer, plan or trust must be the one the excerpt shows.
   - **Visions, dreams, flashbacks** are shown, not told: write "a vision shows ..." or "in a flashback, ...",
     and do not turn them into something a character says or knows unless a character actually says it.
   - Keep details exact (job titles, numbers, whose house, which night); when unsure, be less specific
     rather than guess.
6. **Characters.** Use the names the excerpt uses, in normal case ("Mrs. Christian", not
   "MRS. CHRISTIAN"). A name that only ever appears in capitals or as an acronym (a crewman called
   "ATM") is written as the excerpt writes it. If a name is spelled inconsistently, use the spelling
   that appears most in descriptions and dialogue. At a character's first mention, add a brief role only
   if the excerpt itself establishes it ("Welles, a private investigator,"). Include every character who
   drives the action.
7. **Style.** Third person, present tense, neutral and descriptive, like a plot synopsis. Opening with
   the setting ("In Kubo's cave, ...") or with the protagonist's action are both fine. Do not open with
   or include meta phrases: "Here is", "This excerpt", "In this segment", "The script", "The screenplay",
   "The scene shifts". Paraphrase dialogue; quote at most one short pivotal phrase.
8. **Language.** English.
9. **Noisy source text.** Some excerpts contain OCR errors or mis-tagged lines (a speaker name inside a
   dialogue tag, action text tagged as dialogue). Read through the noise for the intended meaning; never
   copy garbage tokens into the abstract.

## Style examples from the existing CML-Dataset (different movies; style reference only)

> Welles is in his room, watching an 8mm film and using a tape recorder and binoculars to spy on
> Eddie, who is discussing the poor quality of his tapes with a distributor. Welles then calls Eddie,
> threatening him with knowledge of a murder he committed six years ago. Eddie, panicked, calls Dino,
> but Dino dismisses his concerns. Later, Welles visits a bank to secure the 8mm film, researches
> Eddie's phone number, and follows two women to a warehouse in Soho. Inside, he discovers evidence
> linking Dino Velvet to the production of S+M films. Welles calls Max for information on Dino Velvet
> and learns about his specialty in bondage and fetish videos. He also updates Mrs. Christian on his
> progress and requests additional funds.

> James gives Em a mixtape of his favorite sad songs while they wait in line at Adventureland. Em
> confronts two preppy guys who make an anti-Semitic remark, and they leave the line. Later, Em and
> James drive to a beautiful field where Em opens up about her father's possible affair and her trust
> issues. James reassures her, and they share a moment. Back at Adventureland, they almost get caught
> by Em's parents, but manage to escape. Em and James later have a heartfelt conversation about their
> feelings, and Em expresses her fear of losing James. The night ends with them sharing an intimate
> moment in a secluded area behind the arcade.

## Output

For each item, write the file at the item's `abstract_path` (from the batch manifest) as JSON, copying
`content_sha1` from the manifest entry:

```json
{"item_id": "<item_id>", "content_sha1": "<from manifest>", "abstract": "<the abstract text>", "prompt_version": "abstract_v1.5", "author": "<your model name>"}
```

Write only the abstract in `abstract` — no notes, no word counts.
