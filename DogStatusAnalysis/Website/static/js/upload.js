document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const videoInput = document.getElementById('videoInput');
    const submitBtn = document.getElementById('submitBtn');
    const uploadForm = document.getElementById('uploadForm');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressPercent = document.getElementById('progressPercent');

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function isValidFile(file) {
        const validTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime', 
                           'video/x-msvideo', 'video/mkv', 'video/webm', 'video/x-flv'];
        return validTypes.includes(file.type) || 
               /\.(mp4|avi|mov|mkv|wmv|flv|webm)$/i.test(file.name);
    }

    function handleFileSelect(file) {
        if (!isValidFile(file)) {
            alert('Please select a valid video file (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)');
            return;
        }

        if (file.size > 500 * 1024 * 1024) { // 500MB limit
            alert('File size must be less than 500MB');
            return;
        }

        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        
        dropZone.querySelector('.drop-zone-content').classList.add('d-none');
        fileInfo.classList.remove('d-none');
        submitBtn.disabled = false;
    }

    dropZone.addEventListener('click', () => videoInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            videoInput.files = files;
            handleFileSelect(files[0]);
        }
    });

    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    uploadForm.addEventListener('submit', function(e) {
        if (!videoInput.files.length) {
            e.preventDefault();
            alert('Please select a video file first');
            return;
        }

        progressContainer.classList.remove('d-none');
        submitBtn.disabled = true;
        
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            
            progressBar.style.width = progress + '%';
            progressPercent.textContent = Math.round(progress) + '%';
            
            if (progress > 20) {
                document.querySelector('[data-step="upload"]').classList.add('completed');
            }
            if (progress > 40) {
                document.querySelector('[data-step="extract"]').classList.add('completed');
            }
            if (progress > 70) {
                document.querySelector('[data-step="analyze"]').classList.add('completed');
            }
        }, 500);

        window.addEventListener('beforeunload', () => {
            clearInterval(progressInterval);
        });
    });

    function resetForm() {
        dropZone.querySelector('.drop-zone-content').classList.remove('d-none');
        fileInfo.classList.add('d-none');
        submitBtn.disabled = true;
        videoInput.value = '';
        progressContainer.classList.add('d-none');
        progressBar.style.width = '0%';
        progressPercent.textContent = '0%';
        
        document.querySelectorAll('.step').forEach(step => {
            step.classList.remove('completed');
        });
    }

    window.resetUploadForm = resetForm;
});
