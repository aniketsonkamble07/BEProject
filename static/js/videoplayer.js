// Get elements
const video = document.getElementById("videoPlayer");
const sigmlPlayer = document.getElementById("sigmlPlayerContainer");
const playBoth = document.getElementById("playBoth");
const pauseBoth = document.getElementById("pauseBoth");

let isDragging = false, offsetX = 0, offsetY = 0;

// Play/Pause functionality
playBoth.addEventListener("click", () => {
    video.play();
    console.log("SiGML Animation Starts"); // Replace with SiGML play logic
});

pauseBoth.addEventListener("click", () => {
    video.pause();
    console.log("SiGML Animation Paused"); // Replace with SiGML pause logic
});

// Make SiGML Player Draggable
sigmlPlayer.addEventListener("mousedown", (e) => {
    isDragging = true;
    offsetX = e.clientX - sigmlPlayer.getBoundingClientRect().left;
    offsetY = e.clientY - sigmlPlayer.getBoundingClientRect().top;
    sigmlPlayer.style.cursor = "grabbing";
});

document.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    let x = e.clientX - offsetX;
    let y = e.clientY - offsetY;
    sigmlPlayer.style.left = `${x}px`;
    sigmlPlayer.style.top = `${y}px`;
});

document.addEventListener("mouseup", () => {
    isDragging = false;
    sigmlPlayer.style.cursor = "grab";
});

// Handle Fullscreen Mode
document.addEventListener("fullscreenchange", () => {
    if (document.fullscreenElement) {
        video.parentElement.appendChild(sigmlPlayer);
        sigmlPlayer.style.position = "absolute";
        sigmlPlayer.style.bottom = "10px";
        sigmlPlayer.style.right = "10px";
    } else {
        document.body.appendChild(sigmlPlayer);
        sigmlPlayer.style.position = "absolute";
        sigmlPlayer.style.bottom = "50px";
        sigmlPlayer.style.right = "10px";
    }
});
