import React, { useState, useEffect } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [response, setResponse] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        setResponse("File uploaded successfully! 🎉");
      } else {
        const error = await res.text();
        setResponse(`Error uploading file: ${error}`);
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      setResponse("Error uploading file.");
    }
  };

  const handlePreprocessAndDownload = async () => {
    setIsLoading(true);
    setResponse("");
  
    try {
      const res = await fetch("http://127.0.0.1:5000/preprocess", {
        method: "POST",
      });
  
      if (res.ok) {
        // Expect a ZIP file as the response
        const blob = await res.blob(); 
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute("download", "output_prtg.zip"); // Name for the downloaded file
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
  
        setResponse("Preprocessing completed. Files ready for download! 🎉");
      } else {
        const error = await res.text();
        setResponse(`Error during preprocessing: ${error}`);
      }
    } catch (error) {
      console.error("Error during preprocessing:", error);
      setResponse("Error during preprocessing.");
    } finally {
      setIsLoading(false);
    }
  };  

  const handlePredictAndDownload = async () => {
    setIsLoading(true);
    setResponse("");

    try {
        const res = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
        });

if (res.ok) {
        // Expect a ZIP file as the response
        const blob = await res.blob(); 
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute("download", "output_predictions.zip"); // Name for the downloaded file
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
  
        setResponse("Prediction completed. Files ready for download! 🎉");
      } else {
        const error = await res.text();
        setResponse(`Error during prediction: ${error}`);
      }
    } catch (error) {
      console.error("Error during prediction:", error);
      setResponse("Error during prediction.");
    } finally {
      setIsLoading(false);
    }
  };  

  useEffect(() => {
    async function fetchImages() {
      const response = await fetch("http://127.0.0.1:5000/graphs");
      if (response.ok) {
        const imageFiles = await response.json(); // Assuming the backend returns image URLs or filenames
        setImages(imageFiles);
      }
    }
    fetchImages();
  }, []);

  const nextImage = () => {
    setCurrentIndex((prevIndex) => (prevIndex + 1) % images.length);
  };

  const prevImage = () => {
    setCurrentIndex((prevIndex) => (prevIndex - 1 + images.length) % images.length);
  };

  return (
    <div>
      <h1>SLA Prediction App</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={isLoading}>
        Upload File
      </button>
      <button onClick={handlePreprocessAndDownload} disabled={isLoading}>
        Preprocess File
      </button>
      <button onClick={handlePredictAndDownload} disabled={isLoading}>
        Predict File
      </button>
      {isLoading && <p>Processing... Please wait.</p>}
      <p>{response}</p>

      <h2>Graph Viewer</h2>
      <div>
        {images.length > 0 && (
          <>
            <img src={images[currentIndex]} alt={`Event ${currentIndex}`} />
            <div>
              <button onClick={prevImage}>← Previous</button>
              <button onClick={nextImage}>Next →</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
