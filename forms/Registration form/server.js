const express = require('express');
const mongoose = require('mongoose');

const app = express();
const PORT = process.env.PORT || 3000;

// Connect to MongoDB
mongoose.connect('mongodb://127.0.0.1:27017/registration_form', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

// Define the user schema
const userSchema = new mongoose.Schema({
  username: String,
  email: String,
  password: String,
});

// Specify the 'users' collection
const User = mongoose.model('User', userSchema, 'users');

// Set up middleware for parsing JSON
app.use(express.json());

app.use(express.urlencoded({ extended: true }));

// Serve the HTML and CSS files
app.use(express.static('public'));

// Handle form submission
app.post('/register', async (req, res) => {
  try {
    console.log('Received data:', req.body);

    const { username, email, password } = req.body;

    // Basic validation
    if (!username || !email || !password) {
      return res.status(400).send('Missing required fields');
    }

    // Save user to the 'users' collection
    const newUser = new User({ username, email, password });
    await newUser.save();

    res.status(201).send('User registered successfully!');
  } catch (error) {
    console.error(error);
    res.status(500).send('Internal Server Error');
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
