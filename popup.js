// Retrieve data from storage and display it in the popup
chrome.storage.local.get(["myData"], function(result) {
  if (result.myData) {
    // Display the data in the popup
    document.getElementById("data").textContent = result.myData;
  } else {
    // Handle the case when no data is stored
    document.getElementById("data").textContent = "No data found";
  }
});

// Example function to save data to storage
function saveData() {
  var data = document.getElementById("inputData").value;
  
  // Save the data to storage
  chrome.storage.local.set({ "myData": data }, function() {
    // Notify the user that data has been saved
    document.getElementById("message").textContent = "Data saved!";
  });
}

// Example event listener for the save button
document.getElementById("saveButton").addEventListener("click", saveData);
