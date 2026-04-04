// script.js — file input display handler for predict.html
// Updated to match actual element IDs: #fileInput and .file-chosen

document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const fileChosen = document.querySelector('.file-chosen');

    if (fileInput && fileChosen) {
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                fileChosen.textContent = '✔ ' + fileInput.files[0].name;
                fileChosen.style.display = 'block';
            } else {
                fileChosen.textContent = '';
                fileChosen.style.display = 'none';
            }
        });
    }

    // Drag-over highlight for the dropzone
    const dropzone = document.getElementById('dropzone');
    if (dropzone) {
        dropzone.addEventListener('dragover', () => dropzone.classList.add('dragover'));
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', () => dropzone.classList.remove('dragover'));
    }
});