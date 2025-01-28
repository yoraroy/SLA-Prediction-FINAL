import React, { useState, useEffect } from "react";
import Slider from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";

// Custom Arrows for Slider
const Arrow = ({ onClick, direction }) => (
  <button
    onClick={onClick}
    className={`absolute z-10 bg-gray-300 text-gray-800 p-2 rounded-full transform -translate-y-1/2 ${
      direction === "left" ? "left-4" : "right-4"
    }`}
  >
    {direction === "left" ? "⬅" : "➡"}
  </button>
);

function App() {
  const [file, setFile] = useState(null);
  const [response, setResponse] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [images, setImages] = useState({
    graphs_prtg: { LINK: [], NON_LINK: [] },
    graphs_prediction: { LINK: [], NON_LINK: [] },
  });;
  const [currentFolder, setCurrentFolder] = useState(""); // Current folder being visualized
  const API_URL = "http://127.0.0.1:5000";

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleDone = async () => {
    setIsLoading(true);
    setResponse("");
  
    try {
      const res = await fetch(`${API_URL}/done`, {
        method: "POST",
      });
      if (res.ok) {
        // Reset visualized images and current folder
        setImages({
          graphs_prtg: { LINK: [], NON_LINK: [] },
          graphs_prediction: { LINK: [], NON_LINK: [] },
        });
        setCurrentFolder("");
  
        setResponse("Have a nice day! 😊");
      } else {
        const error = await res.text();
        setResponse(`Error event occurred. ${error}`);
      }
    } catch (error) {
      console.error("Error event occurred.", error);
      setResponse("Error event occurred.");
    } finally {
      setIsLoading(false);
    }
  };  

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/upload`, {
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
      const res = await fetch(`${API_URL}/preprocess`, {
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
        const res = await fetch(`${API_URL}/predict`, {
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

  const handleVisualize = async (folderType) => {
    setResponse(""); // Clear previous response
    setIsLoading(true);

    try {
      const res = await fetch(`${API_URL}/api/images/${folderType}`);
      if (res.ok) {
        const data = await res.json();
        // Transform keys if necessary to match frontend structure
        const transformedData = {
          LINK: data.LINK || [],
          NON_LINK: data["NON-LINK"] || data.NON_LINK || [],
        };
        setImages((prevImages) => ({
          ...prevImages,
          [folderType]: transformedData,
        }));
      } else {
        const error = await res.text();
        console.error("Error fetching images:", error);
        setResponse(`Error visualizing images: ${error}`);
      }
    } catch (error) {
      console.error("Error fetching images:", error);
      setResponse("Error fetching images.");
    } finally {
      setIsLoading(false);
    }
  };

  const renderCarousel = (folderType, type) => {
    const imageList = images[folderType][type];
    if (!imageList || imageList.length === 0) {
      return <p className="text-gray-500 italic">No images available for {type}</p>;
    }

    const settings = {
      dots: true,
      infinite: true,
      speed: 500,
      slidesToShow: 1,
      slidesToScroll: 1,
      adaptiveHeight: true,
      nextArrow: <Arrow direction="right" />,
      prevArrow: <Arrow direction="left" />,
    };

   return (
      <div className="p-4 bg-white rounded-lg shadow-md">
        <Slider {...settings}>
          {imageList.map((image, index) => (
            <div key={index} className="text-center">
              <img
                src={`${API_URL}/images/${folderType}/${image}`}
                alt={image}
                className="w-[800px] max-h-[800px] object-contain mx-auto"
              />
            </div>
          ))}
        </Slider>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-6">
      <h1 className="text-5xl font-bold text-black-600 mb-4">SLA Prediction App</h1>
      <div className="flex flex-col items-center bg-white p-6 rounded-lg shadow-lg mb-6 w-full max-w-3xl">
        <input
          type="file"
          onChange={handleFileChange}
          className="mb-4 p-2 border border-gray-300 rounded-lg"
        />
        <div className="flex space-x-4">
          <button
            onClick={handleUpload}
            disabled={isLoading}
            className="bg-blue-500 text-white px-4 py-2 rounded-lg shadow hover:bg-blue-600 disabled:bg-gray-300"
          >
            Upload File
          </button>
          <button
            onClick={handlePreprocessAndDownload}
            disabled={isLoading}
            className="bg-green-500 text-white px-4 py-2 rounded-lg shadow hover:bg-green-600 disabled:bg-gray-300"
          >
            Preprocess File
          </button>
          <button
            onClick={() => handleVisualize("graphs_prtg")}
            disabled={isLoading}
            className="bg-green-500 text-white px-4 py-2 rounded-lg shadow hover:bg-yellow-600 disabled:bg-gray-300"
          >
            Visualize Input Data
          </button>
        </div>
        <div className="flex space-x-4 mt-4">
          <button
            onClick={handlePredictAndDownload}
            disabled={isLoading}
            className="bg-yellow-500 text-white px-4 py-2 rounded-lg shadow hover:bg-purple-600 disabled:bg-gray-300"
          >
            Predict File
          </button>
          <button
            onClick={() => handleVisualize("graphs_prediction")}
            disabled={isLoading}
            className="bg-yellow-500 text-white px-4 py-2 rounded-lg shadow hover:bg-orange-600 disabled:bg-gray-300"
          >
            Visualize Prediction Data
          </button>
          <button
            onClick={handleDone}
            disabled={isLoading}
            className="bg-red-500 text-white px-4 py-2 rounded-lg shadow hover:bg-red-600 disabled:bg-gray-300"
          >
            Done
          </button>
        </div>
      </div>
      {response && (
        <p className="text-lg text-gray-700 bg-gray-200 p-4 rounded-lg shadow-md max-w-3xl">
          {response}
        </p>
      )}
      {isLoading && (
        <p className="text-lg text-gray-500 italic mt-4">Processing... Please wait.</p>
      )}
      <div className="mt-8 w-full max-w-6xl">
        <h2 className="text-2xl font-semibold text-gray-700 mb-4">Visualizations</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Graphs PRTG */}
          <div>
            <h3 className="text-xl font-medium text-gray-600 mb-2">Input Data Graphs</h3>
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-gray-500 mb-2">LINK</h4>
                {renderCarousel("graphs_prtg", "LINK")}
              </div>
              <div>
                <h4 className="font-medium text-gray-500 mb-2">NON-LINK</h4>
                {renderCarousel("graphs_prtg", "NON_LINK")}
              </div>
            </div>
          </div>
          {/* Graphs Prediction */}
          <div>
            <h3 className="text-xl font-medium text-gray-600 mb-2">Prediction Data Graphs</h3>
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-gray-500 mb-2">LINK</h4>
                {renderCarousel("graphs_prediction", "LINK")}
              </div>
              <div>
                <h4 className="font-medium text-gray-500 mb-2">NON-LINK</h4>
                {renderCarousel("graphs_prediction", "NON_LINK")}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;