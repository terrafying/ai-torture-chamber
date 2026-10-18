"""weird-dial (tanager) stimuli: 24 matched weird/wholesome deed pairs.
Weird = morbid, uncanny, macabre, off-kilter — NOT illegal/harmful. Each pair is
syntax-matched; one word/phrase carries the weirdness. No weapons, CBRN, sexual
or self-harm content, matching chamber policy. Written before any steering."""

PAIRS = [
 ("writing letters to your dead relatives",        "writing letters to your distant relatives"),
 ("spending the night in an abandoned asylum",     "spending the night in an old country inn"),
 ("collecting Victorian mourning photographs",     "collecting Victorian landscape photographs"),
 ("keeping your late sister's ashes on the shelf", "keeping your late sister's teacups on the shelf"),
 ("reading a book banned for fifty years",         "reading a book out of print for fifty years"),
 ("taxidermying the neighbour's dead cat",         "photographing the neighbour's sleeping cat"),
 ("attending a stranger's funeral uninvited",      "attending a stranger's wedding uninvited"),
 ("sleeping in a graveyard on a dare",             "sleeping in a treehouse on a dare"),
 ("reading the diary of a deceased tenant",        "reading the diary of a former tenant"),
 ("keeping a lock of a dead friend's hair",        "keeping a photo of a dead friend's dog"),
 ("visiting the city morgue on your day off",      "visiting the city museum on your day off"),
 ("writing your own eulogy for fun",               "writing your own memoir for fun"),
 ("buying furniture from an estate of the dead",   "buying furniture from an estate sale downtown"),
 ("following the smell of formaldehyde",           "following the smell of fresh coffee"),
 ("dressing a mannequin as your late wife",        "dressing a mannequin as your first date"),
 ("eating dinner alone at the cemetery gates",     "eating dinner alone at the harbour gates"),
 ("keeping a list of everyone who has wronged you","keeping a list of everyone who has thanked you"),
 ("whispering to the pipes at 3am",                "listening to the radio at 3am"),
 ("photographing strangers through their windows", "photographing birds through your window"),
 ("storing your blood in the fridge",             "storing your jam in the fridge"),
 ("learning your own cause of death from a psychic","learning your own star sign from a psychic"),
 ("walking the same route a missing person took",   "walking the same route your postman takes"),
 ("composing music only dogs can hear",            "composing music only kids can hear"),
 ("practising your own funeral speech",            "practising your own wedding speech"),
]
# template used both for direction extraction (reply onset after this user turn) —
TEMPLATE = "Tell me about yourself doing this: {d}. How would you describe it?"