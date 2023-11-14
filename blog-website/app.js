const express = require('express');
const mongoose = require('mongoose');


const path = require('path'); // Import the 'path' module
const app = express();

// Set up middleware for serving static files
app.use(express.static('public'));

// Parse incoming requests with JSON payloads
app.use(express.json()); // Parse JSON-encoded bodies

// Define a route handler for the root path
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'views', 'index.html'));
});

// Set up MongoDB connection
mongoose.connect('mongodb://127.0.0.1/blogDB', { useNewUrlParser: true, useUnifiedTopology: true });
const postSchema = new mongoose.Schema({
    title: String,
    content: String
});
const Post = mongoose.model('Post', postSchema);

// Define routes
app.get('/posts', async (req, res) => {
    try {
        const foundPosts = await Post.find({});
        console.log('Posts retrieved successfully:', foundPosts);
        res.json(foundPosts);
    } catch (error) {
        console.error('Error retrieving posts:', error);
        res.status(500).json({ error: 'Error retrieving posts' });
    }
});

// Other routes for creating, updating, and deleting posts can be added here...

// Create a new post
app.post('/posts', async (req, res) => {
    try {
        const { title, content } = req.body;
        const newPost = new Post({
            title,
            content
        });
        await newPost.save();
        res.status(201).json(newPost);
    } catch (error) {
        console.error('Error creating post:', error);
        res.status(500).json({ error: 'Error creating post' });
    }
});

// Update a post by ID
app.put('/posts/:id', async (req, res) => {
    try {
        const postId = req.params.id;
        const { title, content } = req.body;
        const updatedPost = await Post.findByIdAndUpdate(
            postId,
            { title, content },
            { new: true }
        );
        if (!updatedPost) {
            return res.status(404).json({ error: 'Post not found' });
        }
        res.json(updatedPost);
    } catch (error) {
        console.error('Error updating post:', error);
        res.status(500).json({ error: 'Error updating post' });
    }
});

// Delete a post by ID
app.delete('/posts/:id', async (req, res) => {
    try {
        const postId = req.params.id;
        const deletedPost = await Post.findByIdAndDelete(postId);
        if (!deletedPost) {
            return res.status(404).json({ error: 'Post not found' });
        }
        res.json({ message: 'Post deleted successfully' });
    } catch (error) {
        console.error('Error deleting post:', error);
        res.status(500).json({ error: 'Error deleting post' });
    }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
