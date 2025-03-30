const OpenAI = require('openai');

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const transformText = async (req, res) => {
  const { currentMood, desiredTone, originalMessage } = req.body;
  try {
    const prompt = `Current Mood: ${currentMood}\nDesired Tone: ${desiredTone}\nOriginal Message: ${originalMessage}`;
    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        { role: "system", content: "You are an assistant that rewrites messages based on the user's mood and desired tone while preserving their original intent." },
        { role: "user", content: prompt }
      ]
    });
    const transformedMessage = completion.choices[0].message.content;
    res.json({ success: true, data: { transformedMessage } });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ success: false, error: { code: 'TRANSFORM_ERROR', message: 'Failed to transform message.' } });
  }
};

module.exports = { transformText }; 