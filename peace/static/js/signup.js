document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    form.addEventListener('submit', function(event) {
        const email = form.querySelector('#id_email').value;
        const password = form.querySelector('#id_password').value;
        const confirm_password = form.querySelector('#id_confirm_password').value;
        
        // Client-side email format validation
        if (!isValidEmail(email)) {
            alert('Please enter a valid email address.');
            event.preventDefault();
            return false;
        }

        // Client-side password format validation
        if (!isValidPassword(password)) {
            alert('Password must contain a mixture of letters, numbers, and special symbols.');
            event.preventDefault();
            return false;
        }

        // Client-side confirm password validation
        if (password !== confirm_password) {
            alert('Passwords do not match.');
            event.preventDefault();
            return false;
        }
    });

    // Function to validate email format
    function isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    // Function to validate password format
    function isValidPassword(password) {
        const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
        return passwordRegex.test(password);
    }
});
