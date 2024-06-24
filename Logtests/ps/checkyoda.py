# Path to the uploaded YODA file
yoda_file_path = '/Users/user/bassi/LogMoments/tutorials/ps/myshower.yoda'

# Function to inspect the first few lines of the YODA file
def inspect_yoda_file(file_path, num_lines=300):
    with open(file_path, 'r') as file:
        for _ in range(num_lines):
            line = file.readline()
            print(line.strip())

# Inspecting the first few lines of the YODA file to understand its structure
inspect_yoda_file(yoda_file_path)
