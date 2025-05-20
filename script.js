// Define your project data globally or within the function
const projectData = {

  "Learning_Pipeline": {
      github: "https://github.com/vijaydevmasters/Spotify_For_Learning"
  },
  "T5_Navigation": {
      github: "https://github.com/suhasnagaraj99/NLP"
  },
  "Lidar_Alignment": {
      github: "https://github.com/vijaydevmasters/Q-former"
  },
  "Mario_AI": {
      github: "https://github.com/vijaydevmasters/MARIO_DDQN_SWIN"
  },
  "Navigation_YOLO": {
      github: "https://github.com/vijaydevmasters/autonomous_navigation_perception"
  },
  "Airplane_Forecasting": {
      github: "https://github.com/vijaydevmasters/RNN_LSTM_Airline_passanger/tree/main"
  }
};

function openProject(projectName) {
  if (projectData[projectName] && projectData[projectName].github) {
    // Opens the GitHub repo in a new tab
    window.open(projectData[projectName].github, '_blank');
  } else {
    console.error("GitHub link not available for this project.");
  }
}
