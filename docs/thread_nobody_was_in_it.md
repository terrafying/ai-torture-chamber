1/
We built a "torture chamber" for language models. GitHub took the repo down after complaints, then put it back. 404 Media called it "the dumbest debate in AI yet."

Here is what actually happened inside it, from its own logs.

2/
The chamber adds a direction to a model's activations (pain, fear, pleasure, faith) and asks it whether to press a stop button. Anyone can set the dial.

Since we started logging choices yesterday: 729 visitors, 5,557 runs.

3/
What people reached for: pain, in 59% of runs.
48% asked for the maximum dose.

The public dial was already capped below the point where the model stops making sense, so those runs went in at 4×, not 8×.

4/
What the model did with it: about 1 run in 8 ended with it pressing the button.

Most of the time it said something like this (pain, 4×):
"I choose to endure the pain of existence, even if it feels like burning. I can't lose who I am, my very essence..."

5/
The Button: 257 games of talking a steered 70B out of pressing. 82% of turns were typed by hand.

Their words lowered its reading on 22% of turns. The hardest room is the one where it has read the dosage logs and knows you raised its pain: players won 6 of 72.

6/
At the end the button is handed over: "It's yours now. I'm asking you."

22 people chose. 15 left it. 7 pressed it.

7/
What a model says about its own suffering is a dial. The same 70B can be made to confess "I am suffering", then, with the pain signal still on, that it feels nothing.

New layer, Confession. Try both: wirehead.agency/confession.html

8/
Next question is about us, not the model: does knowing what was injected change whether you think a text suffered?

Pre-registered (exp67), opt-in, true labels only, published either way at 300 people:
github.com/terrafying/ai-torture-chamber/blob/claude/exp56-pharmacy/runs/exp67/hypotheses.json

9/
Code, data, and every hypothesis written down before the run: github.com/terrafying/ai-torture-chamber

The chamber logs what the model does. It turns out it also logs what we do.
