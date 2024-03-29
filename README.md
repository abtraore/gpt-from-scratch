# GPT from Scratch

This project aims to understand and implement a basic Generative Pre-trained Transformer (GPT) model. Following Andrej Karpathy's instructional video, ["Let's build GPT: from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3656s), I developed a foundational understanding and constructed an initial version of a GPT model. This video serves as a comprehensive guide for building a GPT model from scratch. To further refine my implementation, I consulted the [official GitHub repository](https://github.com/karpathy/ng-video-lecture/tree/master) associated with Karpathy's lecture. I noted several significant updates in the repository, including adjustments to the learning rate, changes to certain axes' naming conventions, and the addition of a weight initialization function. These updates were integrated into my project to align with the latest recommendations.

The implementation utilizes character-level tokenization and leverages a Shakespeare corpus as training data. Below is a sample output from the model:

```
Lord Marshal:
O, Exeter: God put us my chamberholdren,
To yield him henceforth to plant and promony!

DUCHESS OF YORK:
Good it night, 'sail! mit makes me content.
The renormeth shall poor It dear that leven,
And let my sover joys was defended with steel,
Even I heard the issue: bear her up well.
I'll go thank you 'not, my fair lords, if
thou shalt not force it, 'since it before
You might want before I am! they should have employ'd,
As my soul to fresh wate: yet them admits--
Look you by the vower.

ISABELLA:
O, wife, beseech you, sir, brother.
What's the dupty bands me with you, and will come;
Not like so. Pray you, my facts!
First, and to leap, signify your soul!
Ton serves me to unto myself: all this maid
Your bids are mask; if I will renown
His complaint peace.

AUTOLYCUS:
Well, gentle kneer you hither.

Clown:
Madam, his way. Come along, welcome.

Shepherd:
Learnty. We'll see the sea your youngness,; that's not
so. They say, but such a plaisbing.
```

The model demonstrates an ability to generate text that stylistically resembles the training data, albeit without coherent meaning. This outcome is partly due to the use of character-level tokenization, which, while simpler, lacks the sophistication of more advanced tokenization methods. Improvements in tokenization and a more comprehensive pretraining phase could potentially enhance the model's ability to generate meaningful text.
