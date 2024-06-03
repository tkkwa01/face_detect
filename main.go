package main

import (
	"fmt"
	"gocv.io/x/gocv"
	"image/color"
)

func saveReferenceFace() {
	fmt.Println("Loading reference image...")
	img := gocv.IMRead("reference.jpg", gocv.IMReadColor)
	if img.Empty() {
		fmt.Println("Error reading reference image")
		return
	}
	defer img.Close()

	fmt.Println("Loading cascade classifier...")
	classifier := gocv.NewCascadeClassifier()
	defer classifier.Close()

	if !classifier.Load("haarcascade_frontalface_default.xml") {
		fmt.Println("Error loading cascade file")
		return
	}

	fmt.Println("Detecting faces in reference image...")
	rects := classifier.DetectMultiScale(img)
	if len(rects) == 0 {
		fmt.Println("No faces detected in reference image")
		return
	}

	fmt.Println("Saving detected face...")
	face := img.Region(rects[0])
	gocv.IMWrite("reference_face.jpg", face)

	fmt.Println("Reference face saved")
}

func main() {
	saveReferenceFace()

	fmt.Println("Opening webcam...")
	webcam, err := gocv.VideoCaptureDevice(1)
	if err != nil {
		fmt.Println("Error opening webcam:", err)
		return
	}
	defer webcam.Close()

	fmt.Println("Creating window...")
	window := gocv.NewWindow("Face Recognition")
	defer window.Close()

	fmt.Println("Loading cascade classifier...")
	classifier := gocv.NewCascadeClassifier()
	defer classifier.Close()

	if !classifier.Load("haarcascade_frontalface_default.xml") {
		fmt.Println("Error loading cascade file")
		return
	}

	fmt.Println("Loading reference face image...")
	referenceImg := gocv.IMRead("reference_face.jpg", gocv.IMReadColor)
	if referenceImg.Empty() {
		fmt.Println("Error reading reference face image")
		return
	}
	defer referenceImg.Close()

	fmt.Println("Creating ORB detector...")
	orb := gocv.NewORBWithParams(500, 1.2, 8, 31, 0, 2, 0, 31, 20)
	defer orb.Close()

	fmt.Println("Detecting keypoints in reference face...")
	_, refDescriptors := orb.DetectAndCompute(referenceImg, gocv.NewMat())
	fmt.Printf("Reference Descriptors: Type: %v, Size: %v\n", refDescriptors.Type(), refDescriptors.Size())

	fmt.Println("Starting video capture...")
	img := gocv.NewMat()
	defer img.Close()

	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Println("Cannot read webcam")
			return
		}
		if img.Empty() {
			continue
		}

		rects := classifier.DetectMultiScale(img)
		for _, r := range rects {
			face := img.Region(r)
			_, descriptors := orb.DetectAndCompute(face, gocv.NewMat())
			fmt.Printf("Current Descriptors: Type: %v, Size: %v\n", descriptors.Type(), descriptors.Size())

			if descriptors.Empty() || refDescriptors.Empty() {
				fmt.Println("No descriptors found.")
				continue
			}

			matcher := gocv.NewBFMatcher()
			defer matcher.Close()
			matches := matcher.KnnMatch(refDescriptors, descriptors, 2)

			goodMatches := 0
			for _, m := range matches {
				if len(m) == 2 && m[0].Distance < 0.7*m[1].Distance {
					goodMatches++
				}
			}

			if goodMatches > 15 {
				gocv.Rectangle(&img, r, color.RGBA{0, 255, 0, 0}, 3)
				fmt.Println("Face matched!")
			} else {
				gocv.Rectangle(&img, r, color.RGBA{255, 0, 0, 0}, 3)
				fmt.Println("Face not matched!")
			}
		}

		window.IMShow(img)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}
