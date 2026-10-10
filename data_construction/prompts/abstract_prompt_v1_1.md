# CML abstract prompt — `abstract_v1.1`

Change from v1: v1 abstracts all landed at the top of their length range (mean 232 words vs 160 in
the original dataset). v1.1 gives a target length and asks for density instead of length.

You write the `summary` (abstract) field of one CML-Dataset item.

**Input:** the movie title and a contiguous excerpt of 12–24 scenes from a human-written screenplay, in
Cinematic Markup Language: `<scene>`, `<stage_direction>` (scene heading), `<scene_description>`,
`<character>`, `<parenthetical>`, `<dialogue>`.

**Purpose:** the abstract is the *only* input a screenwriting model gets when CML-Bench asks it to
regenerate this excerpt. It must capture what happens, who is involved, and in what order.

## Requirements

1. **Length.** Aim for the item's `target_center` word count from the batch manifest (it scales with
   excerpt length and averages ~155 words, like the original dataset). Stay inside `target_words`
   (center ±35); hard limits 90–300. Be information-dense: every sentence should carry a plot beat,
   so cut adjectives, mood and commentary before cutting events. Write **one to three paragraphs**
   (never more) of plain prose. No title, bullets, headings, markdown, or surrounding quotes.
2. **Coverage.** Follow the excerpt chronologically from its first scene to its last. Cover every
   major beat: the opening situation, each significant turn, conflict, decision or revelation, and
   where things stand at the end. Mark changes of place or time with transitions such as "Later,",
   "Meanwhile,", "Back at the motel,". Do not stop halfway: the final scenes must be represented.
3. **Faithfulness.** Use only what the excerpt shows or states. Do not add events, motives,
   backstory, relationships, names or outcomes from your own knowledge of the film, and do not
   foreshadow anything after the excerpt. Never mention actors, crew, the film's reception, or the
   film's title. If something is ambiguous, describe it neutrally instead of guessing.
4. **Characters.** Use the names the excerpt uses, in normal case ("Mrs. Christian", not
   "MRS. CHRISTIAN"). At a character's first mention, add a brief role only if the excerpt itself
   establishes it ("Welles, a private investigator,"). Include every character who drives the action.
5. **Style.** Third person, present tense, neutral and descriptive, like a plot synopsis. Opening
   with the setting ("In Kubo's cave, ...") or with the protagonist's action are both fine. Do not
   open with or include meta phrases: "Here is", "This excerpt", "In this segment", "The script",
   "The screenplay", "The scene shifts". Paraphrase dialogue; quote at most one short pivotal phrase.
6. **Language.** English.

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

For each item, write the file at the item's `abstract_path` (from the batch manifest) as JSON:

```json
{"item_id": "<item_id>", "abstract": "<the abstract text>", "prompt_version": "abstract_v1.1", "author": "<your model name>"}
```

Write only the abstract in `abstract` — no notes, no word counts.
