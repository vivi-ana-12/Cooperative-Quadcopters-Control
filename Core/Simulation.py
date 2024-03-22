# from .Quadcopter import Quadcopter
# from .Swarm import Swarm
 
# from ..zmqRemoteApi import RemoteAPIClient

# class Simulation:
#     def __init__(self):

#         self.client = RemoteAPIClient(); # Start RemoteApp connection client
#         self.sim = self.client.getObject('sim');
#         self.client.setStepping(True); # Activate staggered mode
#         print('Connected\n')
        
#         self.swarm = Swarm()
#         self.dataset = [[[] for x in range(13)] for y in range(self.swarm.size)] #Create the dataset with 13 positions for each drone

#     def start_simulation(self):
#         self.sim.startSimulation(); 

        
    
#     def set_object_position(self, obj, world, position):
#         # Define la posición del objeto en la simulación
#         pass
