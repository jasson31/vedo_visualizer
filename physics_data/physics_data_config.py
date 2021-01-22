"""Config file that stores the constants for creating the objects in the simulation"""
PARTICLE_RADIUS = 0.025
MAX_FLUID_START_VELOCITY_XZ = 2.0
MAX_FLUID_START_VELOCITY_Y = 0.5

# default parameters for simulation
default_configuration = {
    "pause": False,
    "stopAt": 16.0,
    "particleRadius": 0.025,
    "numberOfStepsPerRenderUpdate": 1,
    "density0": 1000,
    "simulationMethod": 4,
    "gravitation": [0, -9.81, 0],
    "cflMethod": 0,
    "cflFactor": 1,
    "cflMaxTimeStepSize": 0.005,
    "maxIterations": 100,
    "maxError": 0.01,
    "maxIterationsV": 100,
    "maxErrorV": 0.1,
    "stiffness": 50000,
    "exponent": 7,
    "velocityUpdateMethod": 0,
    "boundaryHandlingMethod": 0,
    "enableDivergenceSolver": True,
    "enablePartioExport": True,
    "enableRigidBodyExport": True,
    "particleFPS": 50.0,
    "partioAttributes": "density;velocity"
}

default_simulation = {
    "contactTolerance": 0.0125,
}

default_fluid = {
    "surfaceTension": 0.2,
    "surfaceTensionMethod": 0,
    "viscosity": 0.01,
    "viscosityMethod": 3,
    "viscoMaxIter": 200,
    "viscoMaxError": 0.05
}

default_rigidbody = {
    "translation": [0, 0, 0],
    "rotationAxis": [0, 1, 0],
    "rotationAngle": 0,
    "scale": [1.0, 1.0, 1.0],
    "color": [0.1, 0.4, 0.6, 1.0],
    "isDynamic": False,
    "isWall": True,
    "restitution": 0.6,
    "friction": 0.0,
    "collisionObjectType": 5,
    "collisionObjectScale": [1.0, 1.0, 1.0],
    "invertSDF": True,
}

default_fluidmodel = {"translation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}


# all fluid model's default scale (particle - 10461)
# box, bunny, dragon, armadillo, sphere, torus
default_fluid_scale = [1.075, 0.857935, 2.2892, 1.17655, 0.6833989, 0.647584]

default_obstacle_size = 10908
