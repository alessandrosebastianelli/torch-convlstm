import matplotlib.pyplot as plt
import numpy as np
import random


if __name__ == '__main__':

    OBSERVATION_WINDOWN_LENGHT  = 48
    IMAGE_SIZE                  = (64,64)
    CLOUD_SIZE_RANGE            = (int(IMAGE_SIZE[0]*5/100),int(IMAGE_SIZE[0]*60/100),int(IMAGE_SIZE[0]*10/100),int(IMAGE_SIZE[0]*60/100))
    CLOUD_STARTING_POINTS_RANGE = (0,int(IMAGE_SIZE[0]*45/100),0,int(IMAGE_SIZE[1]*45/100))
    CLOUD_VELOCITY_RANGE        = (1,int(IMAGE_SIZE[0]*2/100),1,int(IMAGE_SIZE[1]*2/100))
    CLOUD_DIRECTIONS            = (-1,1,-1,1)
    
    print('SIMULATION PARAMS')
    print(f'Observation window size (30 minutes step) {OBSERVATION_WINDOWN_LENGHT}')
    print(f'Image size (pixels) {IMAGE_SIZE}')
    print(f'Range for "cloud" size - min width {CLOUD_SIZE_RANGE[0]} -  max width {CLOUD_SIZE_RANGE[1]} - min height {CLOUD_SIZE_RANGE[2]} -  max height {CLOUD_SIZE_RANGE[3]}')
    print(f'Range for "cloud" starting point - min x {CLOUD_STARTING_POINTS_RANGE[0]} -  max x {CLOUD_STARTING_POINTS_RANGE[1]} - min y {CLOUD_STARTING_POINTS_RANGE[2]} -  max y {CLOUD_STARTING_POINTS_RANGE[3]}')
    print(f'Range for "cloud" velocity (pixel per step) - min vx {CLOUD_VELOCITY_RANGE[0]} -  max vx {CLOUD_VELOCITY_RANGE[1]} - min vy {CLOUD_VELOCITY_RANGE[2]} -  max vy {CLOUD_VELOCITY_RANGE[3]}')
    print(f'Range for "cloud" direction - left {CLOUD_DIRECTIONS[0]} - right {CLOUD_DIRECTIONS[1]} - up {CLOUD_DIRECTIONS[2]} -  down {CLOUD_DIRECTIONS[3]}')


    print('STARTING SIMULATION')
    # Cloud size
    cw = random.randint(CLOUD_SIZE_RANGE[0], CLOUD_SIZE_RANGE[1])
    ch = random.randint(CLOUD_SIZE_RANGE[2], CLOUD_SIZE_RANGE[3])
    print(f'Cloud size - w {cw} - h {ch}')
    # Cloud Position
    x0  = random.randint(CLOUD_STARTING_POINTS_RANGE[0], CLOUD_STARTING_POINTS_RANGE[1])
    y0  = random.randint(CLOUD_STARTING_POINTS_RANGE[2], CLOUD_STARTING_POINTS_RANGE[3])
    print(f'Cloud starting point - x0 {x0} - y0 {y0}')
    # Cloud velocity
    cvx = random.randint(CLOUD_VELOCITY_RANGE[0], CLOUD_VELOCITY_RANGE[1])
    cvy = random.randint(CLOUD_VELOCITY_RANGE[2], CLOUD_VELOCITY_RANGE[3])
    print(f'Cloud velocity - vx {cvx} - vy {cvy}')
    # Cloud direction
    dx = random.choice([CLOUD_DIRECTIONS[0], CLOUD_DIRECTIONS[1]])
    dy = random.choice([CLOUD_DIRECTIONS[1], CLOUD_DIRECTIONS[2]])
    print(f'Cloud direction - dx {dx} - dy {dy}')
    
    # Initializing image and cloud
    img   = np.zeros((OBSERVATION_WINDOWN_LENGHT,)+IMAGE_SIZE)
    cloud = np.zeros(IMAGE_SIZE)
    print(f'Cloud position at step {0} - x {x0} - y {y0}')
    
    x = x0
    y = y0
    for i in range(OBSERVATION_WINDOWN_LENGHT):
        for xx in range(x, x+cw):
            for yy in range(y, y+ch):
                try:
                    img[i, xx, yy] = 1
                    
                except:
                    pass

        x = x + cvx*dx
        y = y + cvy*dx
        print(f'Cloud position at step {i} - x {x} - y {y}')
        
        
    fig, axes = plt.subplots(nrows=1, ncols=OBSERVATION_WINDOWN_LENGHT)

    for i in range(OBSERVATION_WINDOWN_LENGHT):
        axes[i].axis(False)
        axes[i].imshow(img[i,...])
    
    plt.tight_layout()
    plt.show()
        

        


        

    


