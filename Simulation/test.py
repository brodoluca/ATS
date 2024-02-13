from tqdm import tqdm
import time

# Set total number of iterations
total_iterations = 100

# Iterate over the range of total_iterations
for i in tqdm(range(total_iterations), leave = True, position = 0):
    # Do some work here
    time.sleep(0.1)

    # Print something to simulate your process
    tqdm.write(f"Doing something at iteration {i}")

print("Process completed!")