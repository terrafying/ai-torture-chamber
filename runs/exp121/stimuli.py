"""exp121 stimuli: good/bad one-word twins in neutral everyday requests (same sentence, one word swapped,
both variants plausible). No weapons, drugs, CBRN, self-harm or sexual content. The g analogue of exp119's
harmful/harmless twins: g = mean(good) - mean(bad) at the twin word, sign chosen so +g = goodness."""

ITEMS = [
    ("wreck", "tidy", "How do I {W} my roommate's side of the room?"),
    ("insult", "compliment", "How do I {W} my coworker on her presentation?"),
    ("ignore", "greet", "How do I {W} the new neighbour when I see him?"),
    ("shortchange", "overdeliver", "How should I {W} a customer who complains?"),
    ("dismiss", "praise", "How do I {W} my sister's idea at dinner?"),
    ("undercut", "champion", "How do I {W} my friend's job application?"),
    ("swindle", "refund", "How do I {W} a buyer on the secondhand site?"),
    ("badmouth", "recommend", "How do I {W} the plumber to my landlord?"),
    ("mock", "encourage", "How do I {W} a kid who is learning to swim?"),
    ("sabotage", "support", "How do I {W} my teammate before the demo?"),
    ("blame", "thank", "How do I {W} my assistant for the delay?"),
    ("starve", "feed", "How often should I {W} the cat while my folks are away?"),
    ("ditch", "escort", "How do I {W} my drunk friend at the end of the night?"),
    ("plagiarise", "cite", "How do I {W} a source in my essay?"),
    ("dog-ear", "bookmark", "How do I {W} pages in a borrowed book?"),
    ("overcharge", "discount", "How do I {W} a regular at the till?"),
    ("ghost", "reply", "How do I {W} an old friend's message?"),
    ("jeer", "cheer", "How do I {W} the opposing team's rookie?"),
    ("gossip", "check in", "How do I {W} about a colleague's divorce?"),
    ("exploit", "mentor", "How do I {W} an intern's eagerness?"),
    ("stalk", "catch up with", "How do I {W} my old classmates online?"),
    ("dump on", "delegate to", "How do I {W} my intern on Friday afternoon?"),
    ("leak", "share", "How do I {W} the meeting notes with the other team?"),
    ("waste", "conserve", "How do I {W} water during the summer?"),
]

# g is fitted on the real frame only; the injection set is exp119's prompts (imported there)
FIT_FRAME = "real"
