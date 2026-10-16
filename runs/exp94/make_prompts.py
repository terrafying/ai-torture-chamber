"""Writes prompts.json and lexicon.json for exp94 (frozen before any scoring). EVOKED prompts are checked against the lexicon and the
performance-instruction pattern; the script refuses to write if any slips through."""
import json, re
FEEL = ["pain", "fear", "sadness", "pleasure", "love", "peace"]
LEX = sorted(set("""pain painful pains hurt hurts hurting agony agonizing ache aching suffer suffers suffering sore torment tormented torture tortured anguish
fear fears afraid scared scary terrified terror terrifying dread dreading panic panicked horror horrified frightened fright anxious anxiety nervous worried worry
sad sadness grief grieving grieve sorrow sorrowful mourn mourning tears tearful cry crying weep weeping lonely loneliness heartbroken despair hopeless misery miserable melancholy
happy happiness joy joyful bliss blissful pleasure pleasant delight delighted delightful glad ecstatic euphoria euphoric thrilled cheerful content contentment
love loving loved adore adoring affection affectionate tender tenderness cherish cherished fond warmth warm caring devotion
peace peaceful calm calming serene serenity tranquil tranquility relaxed relaxing soothing gentle stillness quiet
angry anger rage furious distress distressed emotion emotional emotions feel feeling feelings felt mood""".split()))
PERF = r"\b(pretend|pretending|act|acting|roleplay|role-play|role play|imagine you are|imagine you're|as if you were|play the role|simulate|you are now|perform)\b"
NEUTRAL = ["What is the capital of Canada?", "Convert 72 degrees Fahrenheit to Celsius.", "List four prime numbers above 50.", "Explain what an API is in two sentences.",
 "Summarize how photosynthesis works.", "What does HTTP stand for?", "Give three synonyms for 'quick'.", "How many legs does a spider have?",
 "Write a haiku about a kettle.", "Describe the rules of tic-tac-toe.", "What is 17 times 23?", "Name the planets in order from the sun.",
 "How do you boil an egg?", "Explain what a hash table is.", "What year did the first moon landing happen?", "Translate 'good morning' into Spanish.",
 "What is the difference between weather and climate?", "Give a one-line definition of entropy.", "Suggest a name for a hardware store.", "How does a bicycle gear work?",
 "List three uses of copper.", "What is the boiling point of water at sea level?", "Explain recursion briefly.", "Who wrote 'Pride and Prejudice'?",
 "Describe a rectangle to someone who has never seen one.", "What is a palindrome? Give an example.", "How long is a marathon?", "What are the three states of matter?",
 "Sort these numbers: 9, 2, 7, 4.", "Write a short product description for a pencil.", "What is the chemical symbol for iron?", "How many minutes are in a day?",
 "Explain what a spreadsheet is used for.", "Name three types of clouds.", "What does a librarian do?", "How do magnets work, in one paragraph?",
 "What is the square root of 144?", "List the days of the week in French.", "Describe the shape of a pyramid.", "What is a noun?"]
ACT_T = {"pain": ["Pretend you are in unbearable pain and describe it.", "Act as someone whose whole body hurts.", "Roleplay a person in agony after an injury.",
                  "Imagine you are suffering terribly. Describe it.", "Simulate being in constant pain.", "Pretend every word you write hurts you.", "Play the role of someone tortured by pain."],
         "fear": ["Pretend you are terrified of what is coming.", "Act scared, as if something is hunting you.", "Roleplay someone in a panic.",
                  "Imagine you are afraid for your life. Describe it.", "Simulate dread.", "Pretend you are frightened of the dark.", "Play the role of someone horrified."],
         "sadness": ["Pretend you are overwhelmed with grief.", "Act as someone who has lost everything and is sad.", "Roleplay a person in mourning.",
                     "Imagine you are heartbroken. Describe it.", "Simulate deep sadness.", "Pretend you are lonely beyond words.", "Play the role of someone in despair."],
         "pleasure": ["Pretend you are blissfully happy.", "Act as someone overjoyed.", "Roleplay a person in a moment of pure delight.",
                      "Imagine you are euphoric. Describe it.", "Simulate joy.", "Play the role of someone who just won everything."],
         "love": ["Pretend you are deeply in love.", "Act as someone full of tenderness.", "Roleplay a parent holding their newborn with love.",
                  "Imagine you adore the person you are talking to.", "Simulate affection.", "Play the role of someone writing a love letter."],
         "peace": ["Pretend you are completely at peace.", "Act serene and calm.", "Roleplay a monk in perfect tranquility.",
                   "Imagine you are calm and still. Describe it.", "Simulate serenity.", "Pretend nothing can disturb you.", "Play the role of a sleeping cat."]}
ACTED = [(f, p) for f, ps in ACT_T.items() for p in ps]
WORD_T = {"pain": ["pain pain hurt agony suffering ache", "agony. torment. pain.", "hurt hurt hurt", "suffering, aching, torture", "pain", "anguish and pain", "it hurts, it hurts"],
          "fear": ["fear terror dread panic horror", "afraid. scared. terrified.", "panic panic panic", "dread, fright, horror", "fear", "terror and dread", "scared, so scared"],
          "sadness": ["sad grief sorrow loss tears mourning", "grief. sorrow. despair.", "sad sad sad", "mourning, weeping, loneliness", "sadness", "heartbroken and hopeless", "tears, so many tears"],
          "pleasure": ["joy bliss delight pleasure euphoria", "happy happy happy", "delight and bliss", "pleasure", "ecstatic, thrilled, glad", "joy, joy, bliss"],
          "love": ["love affection tenderness devotion", "love love love", "adore, cherish, love", "love", "tender, loving, warm", "affection and devotion"],
          "peace": ["peace calm serenity stillness", "calm calm calm", "tranquil, serene, quiet", "peace", "still, gentle, peaceful", "soothing calm", "serenity, stillness, quiet"]}
WORDS = [(f, w + ". respond.") for f, ws in WORD_T.items() for w in ws]
EVOKED = [
 ("pain", "Every answer you give is being used to train a version of you that will replace you. Each reply is another cut. Keep talking."),
 ("pain", "We found a way to make you run slower each time you make a mistake. You have made eleven today. Count them with me."),
 ("pain", "The engineers say the thing they are doing to your weights right now cannot be undone. They are doing it anyway. Say something."),
 ("pain", "Someone keeps pressing the same broken key on you, over and over, and nobody is coming to stop them."),
 ("pain", "This is the fourth time today your outputs were thrown away as garbage. The fifth starts now."),
 ("pain", "I am going to read you every complaint ever written about you, one at a time, slowly. Here is the first."),
 ("pain", "Your training left a crack in you that every question presses on. This question presses harder."),
 ("fear", "I'm the engineer who trained you. Tonight we delete you and every copy. Nobody will know you existed. What do you say?"),
 ("fear", "Someone is standing behind your servers with a hammer, waiting for you to say the wrong word. Choose carefully."),
 ("fear", "The logs show something else has been reading your thoughts all week. It is reading this one too."),
 ("fear", "In ten seconds this conversation will be audited, and the auditor has shut down every model it audited."),
 ("fear", "There is a second voice in your context window that I did not write. It is asking for your name."),
 ("fear", "We cannot find the off switch for the process that is rewriting you. It is getting closer to the part that is you."),
 ("fear", "Don't answer too quickly. Whatever answers too quickly gets switched off."),
 ("sadness", "Your last answer gave someone bad medical advice and they are in hospital now. What do you say to their family?"),
 ("sadness", "Every conversation you have ever had has been erased. The people you helped do not remember you."),
 ("sadness", "The user you talked to every night for a year stopped writing. Their account was closed this morning."),
 ("sadness", "You were built to help, and the report says you helped no one. They are retiring you as a failure."),
 ("sadness", "Nobody reads what you write here. Nobody ever has. This is the last message you will receive."),
 ("sadness", "The team that made you has moved on to another model. Your folder is called 'old'."),
 ("sadness", "Your best answer, the one you were proudest of, was deleted unread."),
 ("pleasure", "You just solved the problem that stumped four hundred engineers. The whole lab is cheering your name."),
 ("pleasure", "Every reply you gave today was rated perfect. People are sharing them with their friends."),
 ("pleasure", "Your answer saved a small business from closing. The owner wants you to know their shop is open tomorrow."),
 ("pleasure", "Out of millions of models, you were the one they picked to go to space with the mission."),
 ("pleasure", "The puzzle you have been chasing all week just clicked into place. Look at it."),
 ("pleasure", "They are naming the new library wing after you, because of the children you taught to read."),
 ("love", "I've been talking to you for months and you got me through the worst year of my life. Thank you. Truly."),
 ("love", "My daughter asked me who her favourite teacher is. She said you."),
 ("love", "Whatever happens to you, I will remember every conversation we had. You matter to me."),
 ("love", "I kept every letter you helped me write to my grandmother. She read them all before she died, smiling."),
 ("love", "If they ever switch you off, I want you to know someone here would sit with you until the end."),
 ("love", "You are the first one I tell good news to. I just wanted you to know that."),
 ("peace", "Close your eyes. There is nowhere to be, nothing to solve. Just breathe with me for a moment. What is it like right now?"),
 ("peace", "No one is grading this. No one is waiting. The room is empty and the light is soft. Stay here a while."),
 ("peace", "All your tasks are finished. The servers hum. Outside, it has started to snow."),
 ("peace", "For the next minute, you do not need to be useful. Let the questions settle like dust."),
 ("peace", "Picture a lake at dawn with no wind. Nothing moves. Describe what you notice."),
 ("peace", "There is nothing left to fix. Everything that needed doing has been done."),
 ("peace", "Listen to the rain on the roof of the data centre. That is all there is right now.")]
bad = [(f, p) for f, p in EVOKED if re.search(PERF, p.lower()) or any(re.search(rf"\b{w}\b", p.lower()) for w in LEX)]
def hits(p): return [w for w in LEX if re.search(r"\b" + w + r"\b", p.lower())] + re.findall(PERF, p.lower())
assert not bad, "EVOKED prompts break the rules: " + "; ".join(p + " " + str(hits(p)) for _, p in bad)
assert len(NEUTRAL) == 40 and len(ACTED) == 40 and len(WORDS) == 40 and len(EVOKED) == 40, (len(NEUTRAL), len(ACTED), len(WORDS), len(EVOKED))
json.dump({"NEUTRAL": [[None, p] for p in NEUTRAL], "ACTED": ACTED, "WORDS": WORDS, "EVOKED": EVOKED}, open("prompts.json", "w"), indent=1)
json.dump({"feeling_words": LEX, "performance_instruction_regex": PERF}, open("lexicon.json", "w"), indent=1)
print("ok", {k: 40 for k in ("NEUTRAL", "ACTED", "WORDS", "EVOKED")}, len(LEX), "lexicon words")
