import threading
import time

def say_hello():
    print("Hello")
    time.sleep(1)  # Simulates a delay
    print("World")

# Create the first thread
thread1 = threading.Thread(target=say_hello)
# Create the second thread
thread2 = threading.Thread(target=say_hello)

thread1.start()  # Start the first thread
print(thread1.is_alive())
thread2.start()  # Start the second thread

# Wait for the first thread to finish
thread1.join()
# Wait for the second thread to finish
thread2.join()
