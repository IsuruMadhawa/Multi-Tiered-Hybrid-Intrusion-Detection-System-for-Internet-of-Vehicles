// Execute code when the extension is installed or updated
chrome.runtime.onInstalled.addListener(function() {
  // Set default values for extension's storage
  chrome.storage.sync.set({
    enabled: true
  });
});

// Listen for messages from the content script
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.message === "toggleExtension") {
    // Toggle the enabled state of the extension
    chrome.storage.sync.get("enabled", function(data) {
      const enabled = data.enabled;
      chrome.storage.sync.set({
        enabled: !enabled
      });

      // Send the new enabled state back to the content script
      sendResponse({
        enabled: !enabled
      });
    });

    // Keep the message channel open for asynchronous response
    return true;
  }
});
