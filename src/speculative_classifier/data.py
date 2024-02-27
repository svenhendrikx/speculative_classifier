TEST_CASES = [
    {"text": "I love this product, it works great!",
     "expected_sentiment": "positive"
     },
    {"text": "This is the worst movie I have ever seen.",
     "expected_sentiment": "negative"
     },
    {"text": "I feel okay about this, nothing special.",
     "expected_sentiment": "neutral"
     },
    {"text": "Absolutely fantastic! Could not have asked for more.",
     "expected_sentiment": "positive"
     },
    {"text": "Terrible service, I'm very disappointed.",
     "expected_sentiment": "negative"
     },
    {"text": "It's just alright, not what I expected but okay.",
     "expected_sentiment": "neutral"
     },
    {"text": "The scenery was breathtaking and unforgettable.",
     "expected_sentiment": "positive"
     },
    {"text": "I hated every moment of this experience.",
     "expected_sentiment": "negative"
     },
    {"text": "I have no strong feelings one way or the other.",
     "expected_sentiment": "neutral"
     },
    {"text": "An outstanding performance, truly a masterpiece.",
     "expected_sentiment": "positive"
     },
    {"text": "The worst customer support experience ever.",
     "expected_sentiment": "negative"
     },
    {"text": "This product does exactly what it's supposed to do.",
     "expected_sentiment": "neutral"
     },
    {"text": "I'm so happy with my purchase, best decision ever!",
     "expected_sentiment": "positive"
     },
    {"text": "I regret buying this, it was a total waste of money.",
     "expected_sentiment": "negative"
     },
    {"text": "Meh, I've seen better, but it's not the worst.",
     "expected_sentiment": "neutral"
     },
    {"text": "A delightful read, I couldn't put it down!",
     "expected_sentiment": "positive"
     },
    {"text": "This app is so buggy, it's practically unusable.",
     "expected_sentiment": "negative"
     },
    {"text": "The movie was okay, nothing to write home about.",
     "expected_sentiment": "neutral"
     },
    {"text": "The food at this restaurant is always amazing!",
     "expected_sentiment": "positive"
     },
    {"text": "I'm so annoyed, my order arrived late and damaged.",
     "expected_sentiment": "negative",
     }
]

PROMPT = """**Sentiment Classification Task**

Given a text snippet, classify the sentiment as positive, neutral, or negative.

**Example 1:**
Text: "I love the new design of the website. It's user-friendly and visually appealing."
Sentiment: Positive

**Example 2:**
Text: "The weather today is average; not too hot, not too cold."
Sentiment: Neutral

**Example 3:**
Text: "I'm disappointed with the service at the restaurant. Our orders were late and cold."
Sentiment: Negative

**Example 4:**
Text: "This book is a masterpiece, filled with intriguing characters and a captivating plot."
Sentiment: Positive

**Example 5:**
Text: "The meeting was scheduled for 10 AM. It started on time."
Sentiment: Neutral

**Example 6:**
Text: "The internet connection is terrible. I can barely load any pages."
Sentiment: Negative

---

**Your Input:**
Text: "{}"
Sentiment:"""
