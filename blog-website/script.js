window.onload = () => {
    const postsContainer = document.getElementById('posts');

    // Fetch and display posts on page load
    fetch('/posts')
        .then(response => response.json())
        .then(posts => {
            displayPosts(posts, postsContainer);
        })
        .catch(error => console.error('Error fetching posts:', error));

    // Handle form submission for creating a new post
    document.getElementById('createPostForm').addEventListener('submit', (event) => {
        event.preventDefault();
        const titleInput = document.getElementById('title');
        const contentInput = document.getElementById('content');

        const title = titleInput.value;
        const content = contentInput.value;

        fetch('/posts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ title, content })
        })
        .then(response => response.json())
        .then(() => {
            // Clear input fields after successfully creating a post
            titleInput.value = '';
            contentInput.value = '';

            // After successfully creating a post, fetch and display all posts
            fetch('/posts')
                .then(response => response.json())
                .then(posts => {
                    displayPosts(posts, postsContainer);
                })
                .catch(error => console.error('Error fetching posts:', error));
        })
        .catch(error => console.error('Error creating post:', error));
    });

    // Handle search functionality
    const searchInput = document.getElementById('searchInput');
    searchInput.addEventListener('input', () => {
        const searchTerm = searchInput.value.toLowerCase();
        fetch('/posts')
            .then(response => response.json())
            .then(posts => {
                const filteredPosts = posts.filter(post =>
                    post.title.toLowerCase().includes(searchTerm) ||
                    post.content.toLowerCase().includes(searchTerm)
                );
                displayPosts(filteredPosts, postsContainer);
            })
            .catch(error => console.error('Error fetching posts:', error));
    });
};

function displayPosts(posts, container) {
    container.innerHTML = '';
    posts.forEach(post => {
        const postElement = document.createElement('div');
        postElement.innerHTML = `
            <h2>${post.title}</h2>
            <p>${post.content}</p>
            <button class="action-button" onclick="editPost('${post._id}')">Edit</button>
            <button class="action-button" onclick="deletePost('${post._id}')">Delete</button>
            <hr>
        `;
        container.appendChild(postElement);
    });
}

function editPost(postId) {
    const updatedTitle = prompt('Enter updated title:');
    const updatedContent = prompt('Enter updated content:');
    
    fetch(`/posts/${postId}`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ title: updatedTitle, content: updatedContent })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(() => window.location.reload())
    .catch(error => console.error('Error updating post:', error));
}

function deletePost(postId) {
    fetch(`/posts/${postId}`, {
        method: 'DELETE'
    })
    .then(() => window.location.reload())
    .catch(error => console.error('Error deleting post:', error));
}
