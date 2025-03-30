require('dotenv').config();
const express = require('express');
const OpenAI = require('openai');
const transformRoutes = require('./routes/transformRoutes');

const app = express();
const port = process.env.PORT || 3000;

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Middleware to parse JSON bodies
app.use(express.json());

// Use transform routes
app.use('/api', transformRoutes);

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
}); 