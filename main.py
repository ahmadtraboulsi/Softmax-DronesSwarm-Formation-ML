import csv
import numpy as np
import matplotlib.pyplot as plt


def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))
def midPoint(x,y):
    return (x[0]+y[0])/2 , (x[1]+y[1])/2
def norm(x):
    return ((x-min(x))/(max(x)-min(x)))

pacc= []

for z in range(100):

    figureId=0

    for t in range(3):


    #data as been recorded at the freq 200 msg/sec
        captureDuration=1
        SimulationsCount=100 #based on data count
    
    #Average Distances in each run TO BE USED AS FEATURES - Need To Normalize
        d1d2All= [0]*SimulationsCount
        d1d3All= [0]*SimulationsCount
        d2d3All= [0]*SimulationsCount

    #Average MidPoints between each of the drones positions TO BE USED AS FEATURES
        d1d2MxAll= [0]*SimulationsCount
        d1d2MyAll= [0]*SimulationsCount
        d1d3MxAll= [0]*SimulationsCount
        d1d3MyAll= [0]*SimulationsCount
        d2d3MxAll= [0]*SimulationsCount
        d2d3MyAll= [0]*SimulationsCount

    #Average linear velocities  TO BE USED AS FEATURES
        d1VxAll = [0]*SimulationsCount
        d1VyAll = [0]*SimulationsCount
        d2VxAll = [0]*SimulationsCount
        d2VyAll = [0]*SimulationsCount
        d3VxAll = [0]*SimulationsCount
        d3VyAll = [0]*SimulationsCount
    
    #Average linear Acceleration
        d1AxAll = [0]*SimulationsCount
        d1AyAll = [0]*SimulationsCount
        d2AxAll = [0]*SimulationsCount
        d2AyAll = [0]*SimulationsCount
        d3AxAll = [0]*SimulationsCount
        d3AyAll = [0]*SimulationsCount

    #simulations
        for i in range(0,SimulationsCount):
            #distances
            d1d2=[0]*int(200*captureDuration)
            d1d3=[0]*int(200*captureDuration)
            d2d3=[0]*int(200*captureDuration)
            #mid points
            d1d2Mx=[0]*int(200*captureDuration)
            d1d2My=[0]*int(200*captureDuration)
            d1d3Mx=[0]*int(200*captureDuration)
            d1d3My=[0]*int(200*captureDuration)
            d2d3Mx=[0]*int(200*captureDuration)
            d2d3My=[0]*int(200*captureDuration)
            #linear velocities
            d1Vx = [0]*int(200*captureDuration)
            d1Vy = [0]*int(200*captureDuration)
            d2Vx = [0]*int(200*captureDuration)
            d2Vy = [0]*int(200*captureDuration)
            d3Vx = [0]*int(200*captureDuration)
            d3Vy = [0]*int(200*captureDuration)
            #linear acceleration
            d1Ax = [0]*int(200*captureDuration)
            d1Ay = [0]*int(200*captureDuration)
            d2Ax = [0]*int(200*captureDuration)
            d2Ay = [0]*int(200*captureDuration)
            d3Ax = [0]*int(200*captureDuration)
            d3Ay = [0]*int(200*captureDuration)
        
            drone1PosArr=None
            drone2PosArr=None
            drone3PosArr=None
    
            if(t==0):
                directory= 'C:/Users/Ahmad/Desktop/TensorFlow/Classification/DronesData/DroneFormationData/RawData/VerticalLineFormationData2/sim' + str(i)
            elif t==1:
                directory= 'C:/Users/Ahmad/Desktop/TensorFlow/Classification/DronesData/DroneFormationData/RawData/TriangleFormationData/sim' + str(i)
            elif t==2:
                directory= 'C:/Users/Ahmad/Desktop/TensorFlow/Classification/DronesData/DroneFormationData/RawData/VerticalLineFormationData/sim' + str(i)
    
                

                

            startIdx=[0]*4
        #Finding the startIndex by taking the minimum startIdx among the three drones thats when the linear velocity becomes above 100 for any of the drones
        ##############
            for j in range(1,4):
               filename= directory +"/navDataFile"+str(j)+".txt"
               #print (filename)
               with open(filename,"r") as csvFile:
                   reader= csv.reader(csvFile, delimiter=',')
                   x=list(reader)
                   result = np.array(x).astype("float")
                   for k in range(len(result)):
                            if  result[k][1]>100:
                                startIdx[j]=k
                                break 
    
            startIdx= startIdx[1:4]
            startIndex=min(startIdx)
            if startIndex==0:
                startIdx.remove(0)
                startIndex=min(startIdx)
            
        
    
        ##############
        
        #Drones 1,2,3
            for j in range(1,4):
                filename= directory +"/navDataFile"+str(j)+".txt"
                #print (filename)
                with open(filename,"r") as csvFile:
                    reader= csv.reader(csvFile, delimiter=',')
                    x=list(reader)
                    result = np.array(x).astype("float")
    
    
            #startIndex should be the time the drone starts moving which is at 665 -PROBLEM StartIndex NOT ALWAYS CORRECT -Fixed as Above
            ##############
    
            #for k in range(len(result)):
            #   if result[k][0]==664:
            #        startIndex=k
            #        break                           
            #stopIndex should be the time the drone has been moving for 2 sec since we receive 200 msg per sec that is 400
                stopIndex=int(200*captureDuration)
            #Delete the remainng rows after stop index from the result
                result=np.delete(result,np.s_[0:startIndex:1],0)
                resultLen= len(result)
                result=np.delete(result,np.s_[stopIndex:resultLen:1],0)
            #Delete all columns except position columns (last two columns)
            #result= np.delete(result,np.s_[0:6:1],1)
                if j==1:
                    drone1PosArr = np.delete(result,np.s_[0:6:1],1)
                    d1Vx = np.delete(np.delete(result,np.s_[0:1:1],1),np.s_[1:8:1],1)
                    d1Vy = np.delete(np.delete(result,np.s_[0:2:1],1),np.s_[1:8:1],1)
                    d1Ax = np.delete(np.delete(result,np.s_[0:3:1],1),np.s_[1:8:1],1)
                    d1Ay = np.delete(np.delete(result,np.s_[0:4:1],1),np.s_[1:8:1],1)
                elif j==2:
                    drone2PosArr = np.delete(result,np.s_[0:6:1],1)
                    d2Vx = np.delete(np.delete(result,np.s_[0:1:1],1),np.s_[1:8:1],1)
                    d2Vy = np.delete(np.delete(result,np.s_[0:2:1],1),np.s_[1:8:1],1)
                    d2Ax = np.delete(np.delete(result,np.s_[0:3:1],1),np.s_[1:8:1],1)
                    d2Ay = np.delete(np.delete(result,np.s_[0:4:1],1),np.s_[1:8:1],1)
                elif j==3:
                    drone3PosArr = np.delete(result,np.s_[0:6:1],1)
                    d3Vx = np.delete(np.delete(result,np.s_[0:1:1],1),np.s_[1:8:1],1)
                    d3Vy = np.delete(np.delete(result,np.s_[0:2:1],1),np.s_[1:8:1],1)
                    d3Ax = np.delete(np.delete(result,np.s_[0:3:1],1),np.s_[1:8:1],1)
                    d3Ay = np.delete(np.delete(result,np.s_[0:4:1],1),np.s_[1:8:1],1)
            
    
            #plt.plot(result[:,0],d1Vx)
            #plt.plot(result[:,0],d1Vy)
            #plt.axis([664,680,-500,1000])
            #plt.show(block=False)
            
            #Compute Euclidian Distance between each of the drones 
            for p in range(0,int(200*captureDuration)):
                d1d2[p]=dist(drone1PosArr[p],drone2PosArr[p])
                d1d3[p]=dist(drone1PosArr[p],drone3PosArr[p])
                d2d3[p]=dist(drone2PosArr[p],drone3PosArr[p])
                #Midpoints
                d1d2Mx[p], d1d2My[p] = midPoint(drone1PosArr[p],drone2PosArr[p])
                d1d3Mx[p], d1d3My[p] = midPoint(drone1PosArr[p],drone3PosArr[p])
                d2d3Mx[p], d2d3My[p] = midPoint(drone2PosArr[p],drone3PosArr[p])
            #Average Linear Velocities
            d1VxAll[i]= np.mean(d1Vx)
            d1VyAll[i]= np.mean(d1Vy)
            d2VxAll[i]= np.mean(d2Vx)
            d2VyAll[i]= np.mean(d2Vy)
            d3VxAll[i]= np.mean(d3Vx)
            d3VyAll[i]= np.mean(d3Vy)
    
            #Average Linear Accelerations
            d1AxAll[i]= np.mean(d1Ax)
            d1AyAll[i]= np.mean(d1Ay)
            d2AxAll[i]= np.mean(d2Ax)
            d2AyAll[i]= np.mean(d2Ay)
            d3AxAll[i]= np.mean(d3Ax)
            d3AyAll[i]= np.mean(d3Ay)
            
    
            #average distances for all runs
            d1d2All[i]=np.mean(d1d2)    
            d1d3All[i]=np.mean(d1d3)
            d2d3All[i]=np.mean(d2d3)
    
            #average of all midpoints
            d1d2MxAll[i]= np.mean(d1d2Mx)
            d1d2MyAll[i]= np.mean(d1d2My)
            d1d3MxAll[i]= np.mean(d1d3Mx)
            d1d3MyAll[i]= np.mean(d1d3My)
            d2d3MxAll[i]= np.mean(d2d3Mx)
            d2d3MyAll[i]= np.mean(d2d3My)
    
    
    
        #DRONE Acceleration PLOTS
        #plt.plot(np.arange(0,captureDuration,captureDuration/len(d3Ay)),d3Ay)
        #plt.axis([0,captureDuration,-2001,2100])
        #plt.xlabel('time', fontsize=18)
        #plt.ylabel('Drone 3 Linear Acceleration X direction', fontsize=16)
        #plt.show(block=False)
    
    
        #DRONE DISTANCES PLOTS
            #plt.figure(figureId)
            #figureId=figureId+1
            #plt.plot(np.arange(0,captureDuration,captureDuration/len(d1d2)),d1d2)
            #plt.axis([0,captureDuration,0.5,5])
            #plt.xlabel('time', fontsize=18)
            #plt.ylabel('distance between drones 1&2', fontsize=16)
            #plt.show(block=False)
    
            #plt.figure(figureId+1)
            #figureId=figureId+1
            #plt.plot(np.arange(0,captureDuration,captureDuration/len(d1d3)),d1d3)
            #plt.axis([0,captureDuration,0.5,5])
            #plt.xlabel('time', fontsize=18)
            #plt.ylabel('distance between drones 1&3', fontsize=16)
            #plt.show(block=False)
    
            #plt.figure(figureId+2)
            #figureId=figureId+1
            #plt.plot(np.arange(0,captureDuration,captureDuration/len(d2d3)),d2d3)
            #plt.axis([0,captureDuration,0.5,5])
            #plt.xlabel('time', fontsize=18)
            #plt.ylabel('distance between drones 2&3', fontsize=16)
            #plt.show(block=False)
        
            #DRONE POSITIONS PLOTS
            #plt.figure(figureId+3)
            #figureId=figureId+1
            #plt.scatter(drone1PosArr[:,0],drone1PosArr[:,1], label="drone1")
            #plt.scatter(drone2PosArr[:,0],drone2PosArr[:,1], label="drone2")
            #plt.scatter(drone3PosArr[:,0],drone3PosArr[:,1], label="drone3")    
            #plt.axis([-1,4,0,9])
            #plt.xticks(np.arange(-1,5,step=1))
            #plt.yticks(np.arange(0,10,step=1))
            #plt.show(block=False)
    
        #Average distances plots during 2 sec
        #plt.figure(figureId+4)
        #figureId=figureId+1
        #plt.plot(d1d2All, label="Avg Dist btw D1&D2")
        #plt.plot(d1d3All, label="Avg Dist btw D1&D3")
        #plt.plot(d2d3All, label="Avg Dist btw D2&D3")
        #plt.xlabel('Simulation No', fontsize=18)
        #plt.ylabel('Average Distances', fontsize=16)
        #plt.legend()
        #plt.show(block=False)
        
        
        #Average linear Velocities plots ( sec duration)
        #plt.figure(figureId+5)
        #figureId=figureId+1
        #plt.scatter(d1VyAll,d1VxAll, label="Avg Linear Velocity Drone 1")
        #plt.scatter(d2VyAll,d2VxAll, label="Avg Linear Velocity Drone 2")
        #plt.scatter(d3VyAll,d3VxAll, label="Avg Linear Velocity Drone 3")
        #plt.xlabel('Average Linear Velocity X Direction', fontsize=18)
        #plt.ylabel('Average Linear Velocity Y Direction', fontsize=16)
        #plt.legend()
        #plt.show(block=False)
        
            
    
        #Average midpoints plots
        #plt.figure(figureId+6)
        #figureId=figureId+1
        #plt.scatter(d1d2MxAll,d1d2MyAll, label="Avg Midpoints btw D1&D2")
        #plt.scatter(d1d3MxAll,d1d3MyAll, label="Avg Midpoints btw D1&D3")
        #plt.scatter(d2d3MxAll,d2d3MyAll, label="Avg Midpoints btw D2&D3")
        #plt.plot(d1d3All, label="Avg Dist btw D1&D3")
        #plt.plot(d2d3All, label="Avg Dist btw D2&D3")
        #plt.axis([-1,4,0,9])
        #plt.xlabel('x-position', fontsize=16)
        #plt.ylabel('y-position', fontsize=16)
        #plt.legend()
        #plt.show(block=False)
    
        #Questions :
        #1) should I normalize among all distances ?
        #2) Midpoint idea ?
        #3) try with and without normalization
    
    
    #DATA : Building the feature vector -
    
    #1. Distances
    #   a. Distance between Drone 1 & 2
    #   b. Distance between Drone 1 & 3
    #   c. Distance between Drone 2 & 3
    
    #2. Linear Velocities (might try relative velocities as well)
    #   a. velocity of drone 1 X
    #   a. velocity of drone 1 Y
    #   b. velocity of drone 2 X
    #   b. velocity of drone 2 Y
    #   c. velocity of drone 3 X
    #   c. velocity of drone 3 Y
    
    #3. Midpoints
    #   a-1. normalized X coordinate between drone 1 and drone 2
    #   a-2. normalized Y coordinate between drone 1 and drone 2
    #   b-1. normalized X coordinate between drone 1 and drone 3
    #   b-2. normalized Y coordinate between drone 1 and drone 3
    #   c-1. normalized X coordinate between drone 2 and drone 3
    #   c-2. normalized Y coordinate between drone 2 and drone 3
    
    
    
        #Normalizing the distances across all drones
        distances= np.hstack((d1d2All,d1d3All,d2d3All))
        normalizedDistances=norm(distances)
        #Split back the distances to d1d2norm d1d3norm d2d3norm
        if(t==0): #Perpendicular Line Formation
            d1d2normV=normalizedDistances[0:1*SimulationsCount]
            d1d3normV=normalizedDistances[1*SimulationsCount:2*SimulationsCount]
            d2d3normV=normalizedDistances[2*SimulationsCount:3*SimulationsCount]
        elif t==1:     #Triangle Formation
            d1d2normT=normalizedDistances[0:1*SimulationsCount]
            d1d3normT=normalizedDistances[1*SimulationsCount:2*SimulationsCount]
            d2d3normT=normalizedDistances[2*SimulationsCount:3*SimulationsCount]
        elif t==2:
            d1d2normV2=normalizedDistances[0:1*SimulationsCount]
            d1d3normV2=normalizedDistances[1*SimulationsCount:2*SimulationsCount]
            d2d3normV2=normalizedDistances[2*SimulationsCount:3*SimulationsCount]
            
            
            
    
        #Normalizing the velocities across all drones
        velocities=np.hstack((d1VxAll,d1VyAll,d2VxAll,d2VyAll,d3VxAll,d3VyAll))
        normalizedVelocities=norm(velocities)
        #Split back the velocities
        if(t==0):
            d1VxnormV=normalizedVelocities[0:1*SimulationsCount]
            d1VynormV=normalizedVelocities[1*SimulationsCount:2*SimulationsCount]
            d2VxnormV=normalizedVelocities[2*SimulationsCount:3*SimulationsCount]
            d2VynormV=normalizedVelocities[3*SimulationsCount:4*SimulationsCount]
            d3VxnormV=normalizedVelocities[4*SimulationsCount:5*SimulationsCount]
            d3VynormV=normalizedVelocities[5*SimulationsCount:6*SimulationsCount]
        elif t==1:
            d1VxnormT=normalizedVelocities[0:1*SimulationsCount]
            d1VynormT=normalizedVelocities[1*SimulationsCount:2*SimulationsCount]
            d2VxnormT=normalizedVelocities[2*SimulationsCount:3*SimulationsCount]
            d2VynormT=normalizedVelocities[3*SimulationsCount:4*SimulationsCount]
            d3VxnormT=normalizedVelocities[4*SimulationsCount:5*SimulationsCount]
            d3VynormT=normalizedVelocities[5*SimulationsCount:6*SimulationsCount]
        elif t==2:
            d1VxnormV2=normalizedVelocities[0:1*SimulationsCount]
            d1VynormV2=normalizedVelocities[1*SimulationsCount:2*SimulationsCount]
            d2VxnormV2=normalizedVelocities[2*SimulationsCount:3*SimulationsCount]
            d2VynormV2=normalizedVelocities[3*SimulationsCount:4*SimulationsCount]
            d3VxnormV2=normalizedVelocities[4*SimulationsCount:5*SimulationsCount]
            d3VynormV2=normalizedVelocities[5*SimulationsCount:6*SimulationsCount]
    
        #Normalizing the midpoints across all drones
        midpointsX= np.hstack((d1d2MxAll, d1d3MxAll, d2d3MxAll))
        midpointsY= np.hstack((d1d2MyAll, d1d3MyAll, d2d3MyAll))
        normalizedMidpointsX=norm(midpointsX)
        normalizedMidpointsY=norm(midpointsY)
        #Split back the midpoints
        if(t==0):        
            d1d2MxnormV= normalizedMidpointsX[0:1*SimulationsCount]
            d1d3MxnormV= normalizedMidpointsX[1*SimulationsCount:2*SimulationsCount]
            d2d3MxnormV= normalizedMidpointsX[2*SimulationsCount:3*SimulationsCount]
            d1d2MynormV= normalizedMidpointsY[0:1*SimulationsCount]
            d1d3MynormV= normalizedMidpointsY[1*SimulationsCount:2*SimulationsCount]
            d2d3MynormV= normalizedMidpointsY[2*SimulationsCount:3*SimulationsCount]
        elif t==1:
            d1d2MxnormT= normalizedMidpointsX[0:1*SimulationsCount]
            d1d3MxnormT= normalizedMidpointsX[1*SimulationsCount:2*SimulationsCount]
            d2d3MxnormT= normalizedMidpointsX[2*SimulationsCount:3*SimulationsCount]
            d1d2MynormT= normalizedMidpointsY[0:1*SimulationsCount]
            d1d3MynormT= normalizedMidpointsY[1*SimulationsCount:2*SimulationsCount]
            d2d3MynormT= normalizedMidpointsY[2*SimulationsCount:3*SimulationsCount]
        elif(t==2):
            d1d2MxnormV2= normalizedMidpointsX[0:1*SimulationsCount]
            d1d3MxnormV2= normalizedMidpointsX[1*SimulationsCount:2*SimulationsCount]
            d2d3MxnormV2= normalizedMidpointsX[2*SimulationsCount:3*SimulationsCount]
            d1d2MynormV2= normalizedMidpointsY[0:1*SimulationsCount]
            d1d3MynormV2= normalizedMidpointsY[1*SimulationsCount:2*SimulationsCount]
            d2d3MynormV2= normalizedMidpointsY[2*SimulationsCount:3*SimulationsCount]
        
        
        #Average distances training_epochsplots during 2 sec
        '''plt.figure(figureId+7)
        #figureId=figureId+1
        if(t==0):
            plt.plot(d1d2normV, label="Normalized Avg Dist btw D1&D2")
            plt.plot(d1d3normV, label="Normalized Avg Dist btw D1&D3")
            plt.plot(d2d3normV, label="Normalized Avg Dist btw D2&D3")
            plt.title("Vertical Line Formation")
        elif t==1:
            plt.title("Triangle Formation")
            plt.plot(d1d2normT, label="Normalized Avg Dist btw D1&D2")
            plt.plot(d1d3normT, label="Normalized Avg Dist btw D1&D3")
            plt.plot(d2d3normT, label="Normalized Avg Dist btw D2&D3")
        elif t==2        :
            plt.plot(d1d2normV2, label="Normalized Avg Dist btw D1&D2")
            plt.plot(d1d3normV2, label="Normalized Avg Dist btw D1&D3")
            plt.plot(d2d3normV2, label="Normalized Avg Dist btw D2&D3")
            plt.title("Vertical Line Formation 2")
            
        plt.xlabel('Simulation No', fontsize=18)
        plt.ylabel('Normalized Average Distances', fontsize=16)
        plt.legend()    
        plt.show(block=False) '''
    
    
        #Average linear Velocities plots ( sec duration)
        '''plt.figure(figureId+8)
            #figureId=figureId+1
            if(t==0):
                plt.title("Vertical Line Formation")
                plt.scatter(d1VynormV,d1VxnormV, label="Normalized Avg Linear Velocity Drone 1")
                plt.scatter(d2VynormV,d2VxnormV, label="Normalized Avg Linear Velocity Drone 2")
                plt.scatter(d3VynormV,d3VxnormV, label="NormalizedAvg Linear Velocity Drone 3")
            elif t==1:
                plt.title("Triangle Formation")
                plt.scatter(d1VynormT,d1VxnormT, label="Normalized Avg Linear Velocity Drone 1")
                plt.scatter(d2VynormT,d2VxnormT, label="Normalized Avg Linear Velocity Drone 2")
                plt.scatter(d3VynormT,d3VxnormT, label="NormalizedAvg Linear Velocity Drone 3")
            elif t==2:
                plt.title("Vertical Line Formation 2")
                plt.scatter(d1VynormV2,d1VxnormV2, label="Normalized Avg Linear Velocity Drone 1")
                plt.scatter(d2VynormV2,d2VxnormV2, label="Normalized Avg Linear Velocity Drone 2")
                plt.scatter(d3VynormV2,d3VxnormV2, label="NormalizedAvg Linear Velocity Drone 3")
                
                
            plt.xlabel('Normalized Average Linear Velocity X Direction', fontsize=18)
            plt.ylabel('Normalized Average Linear Velocity Y Direction', fontsize=16)
            plt.legend()
            plt.show(block=False) '''
    
        
    
        #Average midpoints plots
        '''plt.figure(figureId+9)
        #figureId=figureId+1
        if(t==0):
            plt.title("Vertical Line Formation")
            plt.scatter(d1d2MxnormV,d1d2MynormV, label="Normalized Avg Midpoints btw D1&D2")
            plt.scatter(d1d3MxnormV,d1d3MynormV, label="Normalized Avg Midpoints btw D1&D3")
            plt.scatter(d2d3MxnormV,d2d3MynormV, label="Normalized Avg Midpoints btw D2&D3")
        elif t==1:
            plt.title("Triangle Formation")
            plt.scatter(d1d2MxnormT,d1d2MynormT, label="Normalized Avg Midpoints btw D1&D2")
            plt.scatter(d1d3MxnormT,d1d3MynormT, label="Normalized Avg Midpoints btw D1&D3")
            plt.scatter(d2d3MxnormT,d2d3MynormT, label="Normalized Avg Midpoints btw D2&D3")
        elif t==2:
            plt.title("Vertical Line Formation2")
            plt.scatter(d1d2MxnormV2,d1d2MynormV2, label="Normalized Avg Midpoints btw D1&D2")
            plt.scatter(d1d3MxnormV2,d1d3MynormV2, label="Normalized Avg Midpoints btw D1&D3")
            plt.scatter(d2d3MxnormV2,d2d3MynormV2, label="Normalized Avg Midpoints btw D2&D3")
            
            '''
        #plt.plot(d1d3All, label="Avg Dist btw D1&D3")
        #plt.plot(d2d3All, label="Avg Dist btw D2&D3")
        #plt.axis([-1,4,0,9])
        #plt.xlabel('x-position', fontsize=16)
        #plt.ylabel('y-position', fontsize=16)
        #plt.legend()
        #plt.show(block=False)
        #figureId=figureId+10
    
    #Machine Learning
    # 15 features
    ####################################################
    #RESHAPING THE ARRAYS
    ####################################################
    d1d2normV= np.reshape(d1d2normV, (len(d1d2normV),1))
    d1d3normV= np.reshape(d1d3normV, (len(d1d3normV),1))
    d2d3normV= np.reshape(d2d3normV, (len(d2d3normV),1))
    d1VxnormV= np.reshape(d1VxnormV, (len(d1VxnormV),1))
    d1VynormV= np.reshape(d1VynormV, (len(d1VynormV),1))
    d2VxnormV= np.reshape(d2VxnormV, (len(d2VxnormV),1))
    d2VynormV= np.reshape(d2VynormV, (len(d2VynormV),1))
    d3VxnormV= np.reshape(d3VxnormV, (len(d3VxnormV),1))
    d3VynormV= np.reshape(d3VynormV, (len(d3VynormV),1))
    d1d2MxnormV= np.reshape(d1d2MxnormV, (len(d1d2MxnormV),1))
    d1d2MynormV= np.reshape(d1d2MynormV, (len(d1d2MynormV),1))
    d1d3MxnormV= np.reshape(d1d3MxnormV, (len(d1d3MxnormV),1))
    d1d3MynormV= np.reshape(d1d3MynormV, (len(d1d3MynormV),1))
    d2d3MxnormV= np.reshape(d2d3MxnormV, (len(d2d3MxnormV),1))
    d2d3MynormV= np.reshape(d2d3MynormV, (len(d2d3MynormV),1))
    d1d2normT= np.reshape(d1d2normT, (len(d1d2normT),1))
    d1d3normT= np.reshape(d1d3normT, (len(d1d3normT),1))
    d2d3normT= np.reshape(d2d3normT, (len(d2d3normT),1))
    d1VxnormT= np.reshape(d1VxnormT, (len(d1VxnormT),1))
    d1VynormT= np.reshape(d1VynormT, (len(d1VynormT),1))
    d2VxnormT= np.reshape(d2VxnormT, (len(d2VxnormT),1))
    d2VynormT= np.reshape(d2VynormT, (len(d2VynormT),1))
    d3VxnormT= np.reshape(d3VxnormT, (len(d3VxnormT),1))
    d3VynormT= np.reshape(d3VynormT, (len(d3VynormT),1))
    d1d2MxnormT= np.reshape(d1d2MxnormT, (len(d1d2MxnormT),1))
    d1d2MynormT= np.reshape(d1d2MynormT, (len(d1d2MynormT),1))
    d1d3MxnormT= np.reshape(d1d3MxnormT, (len(d1d3MxnormT),1))
    d1d3MynormT= np.reshape(d1d3MynormT, (len(d1d3MynormT),1))
    d2d3MxnormT= np.reshape(d2d3MxnormT, (len(d2d3MxnormT),1))
    d2d3MynormT= np.reshape(d2d3MynormT, (len(d2d3MynormT),1))
    d1d2normV2= np.reshape(d1d2normV2, (len(d1d2normV2),1))
    d1d3normV2= np.reshape(d1d3normV2, (len(d1d3normV2),1))
    d2d3normV2= np.reshape(d2d3normV2, (len(d2d3normV2),1))
    d1VxnormV2= np.reshape(d1VxnormV2, (len(d1VxnormV2),1))
    d1VynormV2= np.reshape(d1VynormV2, (len(d1VynormV2),1))
    d2VxnormV2= np.reshape(d2VxnormV2, (len(d2VxnormV2),1))
    d2VynormV2= np.reshape(d2VynormV2, (len(d2VynormV2),1))
    d3VxnormV2= np.reshape(d3VxnormV2, (len(d3VxnormV2),1))
    d3VynormV2= np.reshape(d3VynormV2, (len(d3VynormV2),1))
    d1d2MxnormV2= np.reshape(d1d2MxnormV2, (len(d1d2MxnormV2),1))
    d1d2MynormV2= np.reshape(d1d2MynormV2, (len(d1d2MynormV2),1))
    d1d3MxnormV2= np.reshape(d1d3MxnormV2, (len(d1d3MxnormV2),1))
    d1d3MynormV2= np.reshape(d1d3MynormV2, (len(d1d3MynormV2),1))
    d2d3MxnormV2= np.reshape(d2d3MxnormV2, (len(d2d3MxnormV2),1))
    d2d3MynormV2= np.reshape(d2d3MynormV2, (len(d2d3MynormV2),1))
    ##########################################################
    #PREPARING  THE LABELS
    TrainingDataCount=int(0.8*SimulationsCount)
    #Distances only#################################
    #xs_labelV= np.hstack((d1d2normV,d1d3normV,d2d3normV))#,d1VxnormV,d1VynormV,d2VxnormV,d2VynormV,d3VxnormV,d3VynormV,d1d2MxnormV,d1d2MynormV,d1d3MxnormV,d1d3MynormV,d2d3MxnormV,d2d3MynormV))
    #xs_labelT= np.hstack((d1d2normT,d1d3normT,d2d3normT))#,d1VxnormT,d1VynormT,d2VxnormT,d2VynormT,d3VxnormT,d3VynormT,d1d2MxnormT,d1d2MynormT,d1d3MxnormT,d1d3MynormT,d2d3MxnormT,d2d3MynormT))
    #Velocities Only################################
    #xs_labelV= np.hstack((d1VxnormV,d1VynormV,d2VxnormV,d2VynormV,d3VxnormV,d3VynormV))#,d1d2MxnormV,d1d2MynormV,d1d3MxnormV,d1d3MynormV,d2d3MxnormV,d2d3MynormV))
    #xs_labelT= np.hstack((d1VxnormT,d1VynormT,d2VxnormT,d2VynormT,d3VxnormT,d3VynormT))#,d1d2MxnormT,d1d2MynormT,d1d3MxnormT,d1d3MynormT,d2d3MxnormT,d2d3MynormT))
    #Midpoints Only#################################
    #xs_labelV= np.hstack((d1d2MxnormV,d1d2MynormV,d1d3MxnormV,d1d3MynormV,d2d3MxnormV,d2d3MynormV))
    #xs_labelT= np.hstack((d1d2MxnormT,d1d2MynormT,d1d3MxnormT,d1d3MynormT,d2d3MxnormT,d2d3MynormT))
    #Distances and midpoints########################
    #xs_labelV= np.hstack((d1d2normV,d1d3normV,d2d3normV,d1d2MxnormV,d1d2MynormV,d1d3MxnormV,d1d3MynormV,d2d3MxnormV,d2d3MynormV))
    #xs_labelT= np.hstack((d1d2normT,d1d3normT,d2d3normT,d1d2MxnormT,d1d2MynormT,d1d3MxnormT,d1d3MynormT,d2d3MxnormT,d2d3MynormT))
    #Distances and Velocities ########################
    #xs_labelV= np.hstack((d1d2normV,d1d3normV,d2d3normV,d1VxnormV,d1VynormV,d2VxnormV,d2VynormV,d3VxnormV,d3VynormV))
    #xs_labelT= np.hstack((d1d2normT,d1d3normT,d2d3normT,d1VxnormT,d1VynormT,d2VxnormT,d2VynormT,d3VxnormT,d3VynormT))
    #Velocities and Midpoints#########################
    #xs_labelV= np.hstack((d1VxnormV,d1VynormV,d2VxnormV,d2VynormV,d3VxnormV,d3VynormV,d1d2MxnormV,d1d2MynormV,d1d3MxnormV,d1d3MynormV,d2d3MxnormV,d2d3MynormV))
    #xs_labelT= np.hstack((d1VxnormT,d1VynormT,d2VxnormT,d2VynormT,d3VxnormT,d3VynormT,d1d2MxnormT,d1d2MynormT,d1d3MxnormT,d1d3MynormT,d2d3MxnormT,d2d3MynormT))
    #DISTANCE VELOCITIES AND MIDPOINTS################
    xs_labelV= np.hstack((d1d2normV,d1d3normV,d2d3normV,d1VxnormV,d1VynormV,d2VxnormV,d2VynormV,d3VxnormV,d3VynormV,d1d2MxnormV,d1d2MynormV,d1d3MxnormV,d1d3MynormV,d2d3MxnormV,d2d3MynormV))
    xs_labelT= np.hstack((d1d2normT,d1d3normT,d2d3normT,d1VxnormT,d1VynormT,d2VxnormT,d2VynormT,d3VxnormT,d3VynormT,d1d2MxnormT,d1d2MynormT,d1d3MxnormT,d1d3MynormT,d2d3MxnormT,d2d3MynormT))
    
    
    xs=np.vstack((xs_labelV[0:TrainingDataCount],xs_labelT[0:TrainingDataCount]))
    labels= np.matrix([[1.,0.]]*len(d1d2normV[0:TrainingDataCount]) + [[0.,1.]]*len(d1d2normT[0:TrainingDataCount]))
    ##########################################################
    #SHUFFLNIG THE ARRAYS
    arr = np.arange(xs.shape[0])                                               
    np.random.shuffle(arr)
    xs = xs[arr, :]                                                            
    labels = labels[arr, :]
    
    
    #################################
    #PREPARE TEST DATA
    
    #Distances only##################################
    #test_xs_labelV= np.hstack((d1d2normV,d1d3normV,d2d3normV))#,d1VxnormV,d1VynormV,d2VxnormV,d2VynormV,d3VxnormV,d3VynormV,d1d2MxnormV,d1d2MynormV,d1d3MxnormV,d1d3MynormV,d2d3MxnormV,d2d3MynormV))
    #test_xs_labelT= np.hstack((d1d2normT,d1d3normT,d2d3normT))#,d1VxnormT,d1VynormT,d2VxnormT,d2VynormT,d3VxnormT,d3VynormT,d1d2MxnormT,d1d2MynormT,d1d3MxnormT,d1d3MynormT,d2d3MxnormT,d2d3MynormT))
    #test_xs_labelV2=np.hstack((d1d2normV2,d1d3normV2,d2d3normV2))#,d1VxnormV2,d1VynormV2,d2VxnormV2,d2VynormV2,d3VxnormV2,d3VynormV2,d1d2MxnormV2,d1d2MynormV2,d1d3MxnormV2,d1d3MynormV2,d2d3MxnormV2,d2d3MynormV2))
    
    #Velocities Only#################################
    #test_xs_labelV= np.hstack((d1VxnormV,d1VynormV,d2VxnormV,d2VynormV,d3VxnormV,d3VynormV))#,d1d2MxnormV,d1d2MynormV,d1d3MxnormV,d1d3MynormV,d2d3MxnormV,d2d3MynormV))
    #test_xs_labelT= np.hstack((d1VxnormT,d1VynormT,d2VxnormT,d2VynormT,d3VxnormT,d3VynormT))#,d1d2MxnormT,d1d2MynormT,d1d3MxnormT,d1d3MynormT,d2d3MxnormT,d2d3MynormT))
    #test_xs_labelV2=np.hstack((d1VxnormV2,d1VynormV2,d2VxnormV2,d2VynormV2,d3VxnormV2,d3VynormV2))#,d1d2MxnormV2,d1d2MynormV2,d1d3MxnormV2,d1d3MynormV2,d2d3MxnormV2,d2d3MynormV2))
    
    #Midpoints Only##################################
    #test_xs_labelV= np.hstack((d1d2MxnormV,d1d2MynormV,d1d3MxnormV,d1d3MynormV,d2d3MxnormV,d2d3MynormV))
    #test_xs_labelT= np.hstack((d1d2MxnormT,d1d2MynormT,d1d3MxnormT,d1d3MynormT,d2d3MxnormT,d2d3MynormT))
    #test_xs_labelV2=np.hstack((d1d2MxnormV2,d1d2MynormV2,d1d3MxnormV2,d1d3MynormV2,d2d3MxnormV2,d2d3MynormV2))
    
    #Distances and midpoints ########################
    #test_xs_labelV= np.hstack((d1d2normV,d1d3normV,d2d3normV,d1d2MxnormV,d1d2MynormV,d1d3MxnormV,d1d3MynormV,d2d3MxnormV,d2d3MynormV))
    #test_xs_labelT= np.hstack((d1d2normT,d1d3normT,d2d3normT,d1d2MxnormT,d1d2MynormT,d1d3MxnormT,d1d3MynormT,d2d3MxnormT,d2d3MynormT))
    #test_xs_labelV2=np.hstack((d1d2normV2,d1d3normV2,d2d3normV2,d1d2MxnormV2,d1d2MynormV2,d1d3MxnormV2,d1d3MynormV2,d2d3MxnormV2,d2d3MynormV2))
    
    #Distances and Velocities ########################
    #test_xs_labelV= np.hstack((d1d2normV,d1d3normV,d2d3normV,d1VxnormV,d1VynormV,d2VxnormV,d2VynormV,d3VxnormV,d3VynormV))
    #test_xs_labelT= np.hstack((d1d2normT,d1d3normT,d2d3normT,d1VxnormT,d1VynormT,d2VxnormT,d2VynormT,d3VxnormT,d3VynormT))
    #test_xs_labelV2=np.hstack((d1d2normV2,d1d3normV2,d2d3normV2,d1VxnormV2,d1VynormV2,d2VxnormV2,d2VynormV2,d3VxnormV2,d3VynormV2))
    
    #Velocities and Midpoints#########################
    #test_xs_labelV= np.hstack((d1VxnormV,d1VynormV,d2VxnormV,d2VynormV,d3VxnormV,d3VynormV,d1d2MxnormV,d1d2MynormV,d1d3MxnormV,d1d3MynormV,d2d3MxnormV,d2d3MynormV))
    #test_xs_labelT= np.hstack((d1VxnormT,d1VynormT,d2VxnormT,d2VynormT,d3VxnormT,d3VynormT,d1d2MxnormT,d1d2MynormT,d1d3MxnormT,d1d3MynormT,d2d3MxnormT,d2d3MynormT))
    #test_xs_labelV2=np.hstack((d1VxnormV2,d1VynormV2,d2VxnormV2,d2VynormV2,d3VxnormV2,d3VynormV2,d1d2MxnormV2,d1d2MynormV2,d1d3MxnormV2,d1d3MynormV2,d2d3MxnormV2,d2d3MynormV2))
    
    #DISTANCE VELOCITIES AND MIDPOINTS################
    test_xs_labelV= np.hstack((d1d2normV,d1d3normV,d2d3normV,d1VxnormV,d1VynormV,d2VxnormV,d2VynormV,d3VxnormV,d3VynormV,d1d2MxnormV,d1d2MynormV,d1d3MxnormV,d1d3MynormV,d2d3MxnormV,d2d3MynormV))
    test_xs_labelT= np.hstack((d1d2normT,d1d3normT,d2d3normT,d1VxnormT,d1VynormT,d2VxnormT,d2VynormT,d3VxnormT,d3VynormT,d1d2MxnormT,d1d2MynormT,d1d3MxnormT,d1d3MynormT,d2d3MxnormT,d2d3MynormT))
    test_xs_labelV2=np.hstack((d1d2normV2,d1d3normV2,d2d3normV2,d1VxnormV2,d1VynormV2,d2VxnormV2,d2VynormV2,d3VxnormV2,d3VynormV2,d1d2MxnormV2,d1d2MynormV2,d1d3MxnormV2,d1d3MynormV2,d2d3MxnormV2,d2d3MynormV2))
    
    test_xs=np.vstack((test_xs_labelV[TrainingDataCount:],test_xs_labelT[TrainingDataCount:], test_xs_labelV2))
    test_labels= np.matrix([[1.,0.]]*len(d1d2normV[TrainingDataCount:]) + [[0.,1.]]*len(d1d2normT[TrainingDataCount:]) +[[1.,0.]]*len(d1d2normV2))
    
    
    
    #################################
    
    
    train_size, num_features = xs.shape
    
    import tensorflow as tf
    
    learning_rate = 0.01                                                       
    training_epochs = 1000                                                   
    num_labels = 2                                                             
    batch_size = 90                                                           
    
    X = tf.placeholder("float", shape=[None, num_features])                    
    Y = tf.placeholder("float", shape=[None, num_labels])                      
    
    W = tf.Variable(tf.zeros([num_features, num_labels]), name="W")                      
    b = tf.Variable(tf.zeros([num_labels]), name="b")                                    
    y_model = tf.nn.softmax(tf.matmul(X, W) + b)                               
    
    cost = -tf.reduce_sum(Y * tf.log(y_model))                                 
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 
    
    correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(Y, 1))      
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    with tf.Session() as sess:                                                  
        tf.global_variables_initializer().run()                                 
                                                                                
        for step in range(training_epochs * train_size // batch_size):          
            offset = (step * batch_size) % train_size                           
            batch_xs = xs[offset:(offset + batch_size), :]                      
            batch_labels = labels[offset:(offset + batch_size)]                 
            err, _ = sess.run([cost, train_op], feed_dict={X: batch_xs, Y:      
         batch_labels})                                                         
            print (step, err)                                                   
                                                                                
        W_val = sess.run(W)                                                     
        print('w', W_val)                                                       
        b_val = sess.run(b)                                                     
        print('b', b_val)
        writer=tf.summary.FileWriter("./log",sess.graph)
        paccuracy=accuracy.eval(feed_dict={X: test_xs, Y: test_labels})
        pacc.append(paccuracy)
        print("accuracy "+ str(paccuracy))
        print("z="+str(z))
        #using the model to predict output
        #goal1=sess.run(y_model,feed_dict={X:test_xs[19].reshape(1,15)})
        #goal2=sess.run(y_model,feed_dict={X:test_xs[31].reshape(1,15)})
        writer.close()
    
    
    
    #time=[0.3,0.5,1,2]
    #d=[60.75,78.25,95,92.5]
    #v=[100,100,100,100]
    #m=[97.5,97.5,97.5,97.5]
    #dm=[97.5,97.5,97.5,97.5]
    #dv=[100,100,100,100]
    #vm=[97.5,97.5,97.5,97.5]
    #dvm=[97.5,97.5,97.5,97.5]
    #plt.figure(20)
    #plt.title("Accuracy vs Capture Time")
    #plt.plot(time,d, label="Feature:Distance", marker=".")
    #plt.plot(time,v, label="Feature:Velocity", marker="o", linestyle=":")
    #plt.plot(time,m, label="Feature:Midpoints", marker="^", linestyle="-.")
    #plt.plot(time,dm, label="Feature:Distance & Midpoints", marker=">", linestyle="--")
    #plt.plot(time,dv, label="Feature:Distance & Velocity", marker="<", linestyle=":")
    #plt.plot(time,vm, label="Feature:Velocity & Midpoints", marker="8", linestyle="-")
    #plt.plot(time,dvm, label="Feature:Distance, Velocity & Midpoints", marker="p", linestyle=":")
    #plt.legend()
    #plt.show(block=False)
            
