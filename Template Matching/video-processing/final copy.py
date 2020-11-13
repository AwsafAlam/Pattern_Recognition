import cv2
import numpy as np
import time

#read the reference image in gray Scale
referenceImage = cv2.imread("reference.jpg" , 0)
shapeOfReferenceImage = referenceImage.shape

M=shapeOfReferenceImage[0]
N= shapeOfReferenceImage[1]

#set the value of p
p=4

#-----------------------------------------------------------------------------------
#read the video
frames=[]

# Capture the frames of the input video to an array of frames
cap = cv2.VideoCapture("input.mov")
fps = cap.get(cv2.CAP_PROP_FPS)

flag = 1

while flag:
    flag,frame = cap.read()

    if(flag == 0):
        break

    grayFrame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    frames.append(grayFrame)

cap.release()
numOfFrames = len(frames)
#-----------------------------------------------------------------------------------


def exhaustiveSearch():
    global frames, numOfFrames

    print("Working on Exhaustive searching")

    startingTime = time.time()
    outputFrames = []

    exhaustiveSearching = 0

    # exhaustive working with the first frame of the video
    firstFrameShape = frames[0].shape

    I = firstFrameShape[0]
    J = firstFrameShape[1]

    prevI = -1
    PrevJ = -1

    # working of the 1st frame
    minD = np.inf
    markedIIndex = 0
    markedJIndex = 0

    iStart = 0
    iEnd = I - M + 1
    jStart = 0
    jEnd = J - N + 1

    frame = frames[0]

    for i in range(iStart, iEnd):
        for j in range(jStart, jEnd):
            # calculate the D(m,n)
            # first make a sub matrix
            subMatrix = frame[i:i + M, j:j + N]
            subMatrix = subMatrix.astype(np.int64)

            # calculation of difference
            tempRef = referenceImage.astype(np.int64)
            difference = tempRef - subMatrix
            difference = np.absolute(difference)

            diffSquare = difference * difference

            d = np.sum(diffSquare)

            exhaustiveSearching += 1

            if (d < minD):
                minD = d
                markedIIndex = i
                markedJIndex = j

    # update the prevI and prevJ
    prevI = markedIIndex
    prevJ = markedJIndex

    # convert the frame to rgb
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # draw the rectangle around the reference image
    cv2.rectangle(rgbFrame, (markedJIndex, markedIIndex), (markedJIndex + N, markedIIndex + M), (0, 0, 255), 2)
    # append to the output frame
    outputFrames.append(rgbFrame)

    # now working with the other frames
    for index in range(1, numOfFrames, 1):

        iStart = prevI - p
        iEnd = prevI + p
        jStart = prevJ - p
        jEnd = prevJ + p

        minD = np.inf
        markedIIndex = 0
        markedJIndex = 0

        frame = frames[index]

        for i in range(iStart, iEnd):
            for j in range(jStart, jEnd):
                if ((i < 0) or (i > I - M) or (j < 0) or (j > J - N)):
                    continue

                # calculate the D(m,n)
                # first make a sub matrix
                subMatrix = frame[i:i + M, j:j + N]
                subMatrix = subMatrix.astype(np.int64)

                # calculation of difference
                tempRef = referenceImage.astype(np.int64)
                difference = tempRef - subMatrix
                difference = np.absolute(difference)

                diffSquare = difference * difference

                d = np.sum(diffSquare)

                exhaustiveSearching += 1

                if (d < minD):
                    minD = d
                    markedIIndex = i
                    markedJIndex = j

        # update the prevI and prevJ
        prevI = markedIIndex
        prevJ = markedJIndex

        # convert the frame to rgb
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # draw the rectangle around the reference image
        cv2.rectangle(rgbFrame, (markedJIndex, markedIIndex), (markedJIndex + N, markedIIndex + M), (0, 0, 255), 2)
        # append to the output frame
        outputFrames.append(rgbFrame)

    endingTime = time.time()

    timeDuration = endingTime - startingTime

    # now make the output video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output = cv2.VideoWriter("final_Exhaustive.mov", fourcc, fps, (J, I))

    for frame in outputFrames:
        output.write(frame)

    output.release()

    print("Exhasutive search")
    print("-----------------")
    print("Value of P: ", p)
    print("Time Taken: ", timeDuration, "seconds")
    print("Searching Done: ", exhaustiveSearching)
    print("Avg Searching Per Frame: ", exhaustiveSearching / numOfFrames)
    print()

def twoDLogSearch():
    global frames, numOfFrames

    print("Working on 2D Log Searching")

    startingTime = time.time()
    outputFrames = []

    twoDLogSearching = 0

    # exhasutive working with the first frame of the video
    firstFrameShape = frames[0].shape

    I = firstFrameShape[0]
    J = firstFrameShape[1]

    prevI = -1
    PrevJ = -1

    # working of the 1st frame
    minD = np.inf
    markedIIndex = 0
    markedJIndex = 0

    iStart = 0
    iEnd = I - M + 1
    jStart = 0
    jEnd = J - N + 1

    frame = frames[0]

    for i in range(iStart, iEnd):
        for j in range(jStart, jEnd):
            # calculate the D(m,n)
            # first make a sub matrix
            subMatrix = frame[i:i + M, j:j + N]
            subMatrix = subMatrix.astype(np.int64)

            # calculation of difference
            tempRef = referenceImage.astype(np.int64)
            difference = tempRef - subMatrix
            difference = np.absolute(difference)

            diffSquare = difference * difference

            d = np.sum(diffSquare)

            if (d < minD):
                minD = d
                markedIIndex = i
                markedJIndex = j

    # update the prevI and prevJ
    prevI = markedIIndex
    prevJ = markedJIndex

    # convert the frame to rgb
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # draw the rectangle around the reference image
    cv2.rectangle(rgbFrame, (markedJIndex, markedIIndex), (markedJIndex + N, markedIIndex + M), (0, 0, 255), 2)
    # append to the output frame
    outputFrames.append(rgbFrame)

    # working with other frames

    for index in range(1, numOfFrames, 1):

        prevP = p
        frame = frames[index]

        while (True):
            # determine the value of k and d
            k = np.ceil(np.log2(prevP))
            distance = int(np.power(2, k - 1))

            # now determine the 9 points
            points = {
                "1": [prevI, prevJ],
                "2": [prevJ - distance, prevJ],
                "3": [prevI - distance, prevJ + distance],
                "4": [prevI - distance, prevJ - distance],
                "5": [prevI + distance, prevJ],
                "6": [prevI + distance, prevJ + distance],
                "7": [prevI + distance, prevJ - distance],
                "8": [prevI, prevJ + distance],
                "9": [prevI, prevJ - distance]
            }

            # now work with these 9 points only
            # find the point with the minimum distance

            indexKey = 0
            minD = np.inf

            for key in points:

                i = points[key][0]
                j = points[key][1]

                if ((i < 0) or (i > I - M) or (j < 0) or (j > J - N)):
                    continue

                # calculate the D(m,n)
                # first make a sub matrix
                subMatrix = frame[i:i + M, j:j + N]
                subMatrix = subMatrix.astype(np.int64)

                # calculation of difference
                tempRef = referenceImage.astype(np.int64)
                difference = tempRef - subMatrix
                difference = np.absolute(difference)

                diffSquare = difference * difference

                d = np.sum(diffSquare)

                twoDLogSearching += 1

                if (d < minD):
                    minD = d
                    indexKey = key

            # point found. set the point as the prevI and prevJ
            prevI = points[indexKey][0]
            prevJ = points[indexKey][1]
            prevP = prevP / 2

            # when the distance is 1, break the while loop
            if (distance == 1):
                break

        # convert the frame to rgb
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # draw the rectangle around the reference image
        cv2.rectangle(rgbFrame, (prevJ, prevI), (prevJ + N, prevI + M), (0, 0, 255), 2)
        # append to the output frame
        outputFrames.append(rgbFrame)

    endingTime = time.time()

    timeDuration = endingTime - startingTime

    # now make the output video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output = cv2.VideoWriter("final_2DLog.mov", fourcc, fps, (J, I))

    for frame in outputFrames:
        output.write(frame)

    output.release()

    print("2D Log search")
    print("-----------------")
    print("Value of P: ", p)
    print("Time Taken: ", timeDuration, "seconds")
    print("Searching Done: ", twoDLogSearching)
    print("Avg Searching Per Frame: " , twoDLogSearching/numOfFrames)
    print()

def hierarchicalSearch():
    global frames, numOfFrames
    print("working on Hierarchical search")

    startingTime = time.time()
    outputFrames = []

    hierarchicalSearching = 0

    # exhaustive working with the first frame of the video
    firstFrameShape = frames[0].shape

    I = firstFrameShape[0]
    J = firstFrameShape[1]

    prevI = -1
    PrevJ = -1

    # working of the 1st frame
    minD = np.inf
    markedIIndex = 0
    markedJIndex = 0

    iStart = 0
    iEnd = I - M + 1
    jStart = 0
    jEnd = J - N + 1

    frame = frames[0]

    for x in range(iStart, iEnd):
        for y in range(jStart, jEnd):
            # calculate the D(m,n)
            # first make a sub matrix
            subMatrix = frame[x:x + M, y:y + N]
            subMatrix = subMatrix.astype(np.int64)

            # calculation of difference
            tempRef = referenceImage.astype(np.int64)
            difference = tempRef - subMatrix
            difference = np.absolute(difference)

            diffSquare = difference * difference

            d = np.sum(diffSquare)

            if (d < minD):
                minD = d
                markedIIndex = x
                markedJIndex = y

    # update the prevI and prevJ
    prevI = markedIIndex
    prevJ = markedJIndex

    # convert the frame to rgb
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # draw the rectangle around the reference image
    cv2.rectangle(rgbFrame, (markedJIndex, markedIIndex), (markedJIndex + N, markedIIndex + M), (0, 0, 255), 2)
    # append to the output frame
    outputFrames.append(rgbFrame)

    #working with other frames
    for index in range(1, numOfFrames ,1):
        frame = frames[index]

        #-----------------------------------------------------------------------------------------------
        #step 1
        #create level 0,1,2 images
        referenceImageLevels =[]
        testImageLevels =[]

        #level 0. The original image
        referenceImageLevels.append(referenceImage)
        testImageLevels.append(frame)

        #level 1
        referenceImageLevels.append(cv2.pyrDown(referenceImageLevels[0]))
        testImageLevels.append(cv2.pyrDown(testImageLevels[0]))

        # level 2
        referenceImageLevels.append(cv2.pyrDown(referenceImageLevels[1]))
        testImageLevels.append(cv2.pyrDown(testImageLevels[1]))
        # -----------------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------------
        #step 2
        tempP = p//4
        centreI = prevI//4
        centreJ =prevJ//4

        minD = np.inf
        markedIIndex = 0
        markedJIndex = 0

        tempM= referenceImageLevels[2].shape[0]
        tempN= referenceImageLevels[2].shape[1]
        tempI= testImageLevels[2].shape[0]
        tempJ= testImageLevels[2].shape[1]

        pointsX = [centreJ-tempP , centreJ , centreJ+tempP]
        pointsY = [centreI-tempP , centreI , centreI+tempP]

        for y in pointsY:
            for x in pointsX:
                if y < 0 or y > tempI-tempM or x < 0 or x > tempJ - tempN:
                    continue
                # calculate the D(m,n)
                # first make a sub matrix
                subMatrix = testImageLevels[2][y:y + tempM, x:x + tempN]
                subMatrix = subMatrix.astype(np.int64)

                # calculation of difference
                tempRef = referenceImageLevels[2].astype(np.int64)
                difference = tempRef - subMatrix
                difference = np.absolute(difference)

                diffSquare = difference * difference

                d = np.sum(diffSquare)

                hierarchicalSearching += 1

                if (d < minD):
                    minD = d
                    markedIIndex = y
                    markedJIndex = x

        y1 = markedIIndex
        x1 = markedJIndex
        # -----------------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------------
        # step 3
        tempP = 1
        centreI = int( 2*y1)
        centreJ = int( 2*x1)

        #centreI = int( (prevI/2) + (2*y1) )
        #centreJ = int( (prevJ/2) + (2*x1) )

        minD = np.inf
        markedIIndex = 0
        markedJIndex = 0

        tempM = referenceImageLevels[1].shape[0]
        tempN = referenceImageLevels[1].shape[1]
        tempI = testImageLevels[1].shape[0]
        tempJ = testImageLevels[1].shape[1]

        pointsX = [centreJ - tempP, centreJ, centreJ + tempP]
        pointsY = [centreI - tempP, centreI, centreI + tempP]

        for y in pointsY:
            for x in pointsX:
                if y < 0 or y > tempI - tempM or x < 0 or x > tempJ - tempN:
                    continue
                    # calculate the D(m,n)
                # first make a sub matrix
                subMatrix = testImageLevels[1][y:y + tempM, x:x + tempN]
                subMatrix = subMatrix.astype(np.int64)

                # calculation of difference
                tempRef = referenceImageLevels[1].astype(np.int64)

                difference = tempRef - subMatrix
                difference = np.absolute(difference)

                diffSquare = difference * difference

                d = np.sum(diffSquare)

                hierarchicalSearching += 1

                if (d < minD):
                    minD = d
                    markedIIndex = y
                    markedJIndex = x

        y2 = markedIIndex
        x2 = markedJIndex
        # -----------------------------------------------------------------------------------------------
        #
        # -----------------------------------------------------------------------------------------------
        # step 4
        tempP = 1
        centreI = int (2*y2)
        centreJ = int (2*x2)

        #centreI = int ( prevI + (2*y2) )
        #centreJ = int ( prevJ + (2*x2) )

        minD = np.inf
        markedIIndex = 0
        markedJIndex = 0

        tempM = referenceImageLevels[0].shape[0]
        tempN = referenceImageLevels[0].shape[1]
        tempI = testImageLevels[0].shape[0]
        tempJ = testImageLevels[0].shape[1]

        pointsX = [centreJ - tempP, centreJ, centreJ + tempP]
        pointsY = [centreI - tempP, centreI, centreI + tempP]

        for y in pointsY:
            for x in pointsX:
                if y < 0 or y > tempI - tempM or x < 0 or x > tempJ - tempN:
                    continue

                # calculate the D(m,n)
                # first make a sub matrix
                subMatrix = testImageLevels[0][y:y + tempM, x:x + tempN]
                subMatrix = subMatrix.astype(np.int64)

                # calculation of difference
                tempRef = referenceImageLevels[0].astype(np.int64)
                difference = tempRef - subMatrix
                difference = np.absolute(difference)

                diffSquare = difference * difference

                d = np.sum(diffSquare)

                hierarchicalSearching += 1

                if (d < minD):
                    minD = d
                    markedIIndex = y
                    markedJIndex = x
        # -----------------------------------------------------------------------------------------------

        # update the prevI and prevJ
        prevI = markedIIndex
        prevJ = markedJIndex

        # convert the frame to rgb
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # draw the rectangle around the reference image
        cv2.rectangle(rgbFrame, (markedJIndex, markedIIndex), (markedJIndex + N, markedIIndex + M), (0, 0, 255), 2)
        # append to the output frame
        outputFrames.append(rgbFrame)


    endingTime = time.time()

    timeDuration = endingTime - startingTime

    # now make the output video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output = cv2.VideoWriter("final_Hierarchical.mov", fourcc, fps, (J, I))

    for frame in outputFrames:
        output.write(frame)

    output.release()

    print("Hierarchical search")
    print("-----------------")
    print("Value of P: ", p)
    print("Time Taken: ", timeDuration, "seconds")
    print("Searching Done: ", hierarchicalSearching)
    print("Avg. Searching Per Frame: " , hierarchicalSearching/numOfFrames)
    print()


def main():
    exhaustiveSearch()
    twoDLogSearch()
    hierarchicalSearch()


if __name__ == "__main__":
    main()


