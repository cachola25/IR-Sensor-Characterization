import tkinter as tk
from tkinter import messagebox, filedialog

robot_name = None   # Get the robot name to connect form
selected_zones = [] # Get the zones the robot is in
filename = None     # Get the CSV file for the data
num_rows = 500      # Get the number of rows to collect

# Function to move from Screen 1 to Screen 2
def continue_from_screen1():
    global robot_name
    selected_option = robot_var.get()
    if selected_option == "Other":
        robot_name = other_entry.get().strip()
        print(robot_name)
        if not robot_name:
            messagebox.showerror("Input Error", "Please enter the robot name.")
            return
    else:
        robot_name = selected_option

    screen1.destroy()
    show_screen2()

# Function to move from Screen 2 to Screen 3
def continue_from_screen2():
    # if not selected_zones:
        # messagebox.showerror("Selection Error", "Please select at least one zone.")
        # return

    screen2.destroy()
    show_screen3()

# Function to move from Screen 3 to Screen 4
def continue_from_screen3():
    global filename
    selected_option = file_option_var.get()
    # If the user includes the file extension, make sure it's a CSV
    if selected_option == "New File":
        entered_filename = filename_entry.get().strip()
        if not entered_filename:
            messagebox.showerror("Input Error", "Please enter a filename.")
            return
    
        if '.' not in entered_filename:
            entered_filename += '.csv'
        else:
        
            ext = entered_filename.split('.')[-1]
            if ext.lower() != 'csv':
                messagebox.showerror("Invalid Extension", "Please enter a filename with a '.csv' extension.")
                return
        filename = entered_filename
    elif selected_option == "Existing File":
        # Double check that CSV was selected
        if not selected_file_label.cget("text"):
            messagebox.showerror("Input Error", "Please select an existing '.csv' file.")
            return
        filename = selected_file_label.cget("text")
    else:
        messagebox.showerror("Selection Error", "Please select an option.")
        return

    screen3.destroy()
    show_screen4()

# Return from the program when all options are defined and print the options to the screen
def continue_from_screen4():
    global num_rows
    selected_option = rows_option_var.get()
    if selected_option == "Default":
        num_rows = 500
    elif selected_option == "Custom":
        entered_rows = rows_entry.get().strip()
        if not entered_rows:
            messagebox.showerror("Input Error", "Please enter the number of rows.")
            return
        try:
            num_rows = int(entered_rows)
            if num_rows <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a positive integer for the number of rows.")
            return
    else:
        messagebox.showerror("Selection Error", "Please select an option.")
        return


    print(f"Selected Robot Name: {robot_name}")
    print(f"Selected Zones: {selected_zones}")
    print(f"Selected Filename: {filename}")
    print(f"Number of Rows: {num_rows}")
    screen4.destroy()

# Function that will open a file explorer for the user to select the data file
def browse_file():
    # Only allow CSVs to be selected
    file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv")],
        defaultextension=".csv"
    )
    if file_path:
        selected_file_label.config(text=file_path)

# Make the GUI behave differently based on what option is selected
def on_file_option_change(*args):
    selected_option = file_option_var.get()

    if selected_option == "New File":
        # Disable the browse button and enable the text area
        filename_entry.configure(state='normal')
        browse_button.configure(state='disabled')
    elif selected_option == "Existing File":
        # Disable the text area and enable the browse button
        filename_entry.configure(state='disabled')
        browse_button.configure(state='normal')
    else:
        # Unfocus the window if there is an error
        filename_entry.configure(state='disabled')
        browse_button.configure(state='disabled')

# Make the GUI behave differently based on what option is selected
def on_rows_option_change(*args):
    selected_option = rows_option_var.get()
    if selected_option == "Default":
        # Disable the text area
        rows_entry.configure(state='disabled')
    elif selected_option == "Custom":
        # Enable the text area
        rows_entry.configure(state='normal')
    else:
        rows_entry.configure(state='disabled')

# Function that keeps track of which zones the user has selected
def toggle_zone(zone):
    if zone in selected_zones:
        selected_zones.remove(zone)
        zone_buttons[zone].configure(bg="SystemButtonFace", relief="raised", fg="black")
    else:
        selected_zones.append(zone)
        zone_buttons[zone].configure(bg="dodgerblue", relief="sunken", fg="white")
        
    # Define the sorting key function
    def zone_sort_key(z):
        col = z[0]
        row = int(z[1:])
        col_num = ord(col) - ord('a')  # Convert column letter to a number (0 for 'a', 1 for 'b', etc.)
        return (col_num, row)

    # Sort the selected zones
    sorted_zones = sorted(selected_zones, key=zone_sort_key)
    selected_zones_label.config(text=f"Selected Zones: {', '.join(sorted_zones)}")

# Screen to get the name of the robot
def show_screen1():
    global screen1, robot_var, other_entry

    screen1 = tk.Tk()
    screen1.title("Robot Selection")

    screen1.attributes('-topmost', True)
    screen1.update()
    
    instruction_label = tk.Label(screen1, text="Select the name of the robot:")
    instruction_label.pack(pady=10)

    # Initialize robot_var with the default value (you can set it to "CapstoneRobot1" or "Roomba")
    robot_var = tk.StringVar(value="CapstoneRobot1")

    # First Robot
    capstone_option = tk.Radiobutton(screen1, text="CapstoneRobot1", variable=robot_var, value="CapstoneRobot1")
    capstone_option.pack(anchor='w', padx=20)

    # Second Robot
    roomba_option = tk.Radiobutton(screen1, text="Roomba", variable=robot_var, value="Roomba")
    roomba_option.pack(anchor='w', padx=20)

    # Enter a name for the other robot
    other_option = tk.Radiobutton(screen1, text="Other", variable=robot_var, value="Other")
    other_option.pack(anchor='w', padx=20)

    other_entry = tk.Entry(screen1)
    other_entry.pack(pady=5, padx=20, fill='x')
    other_entry.configure(state='disabled')

    # Disable text area depending on option selected
    def on_robot_selection(*args):
        if robot_var.get() == "Other":
            other_entry.configure(state='normal')
        else:
            other_entry.configure(state='disabled')
            other_entry.delete(0, tk.END)

    robot_var.trace_add('write', on_robot_selection)
    continue_button = tk.Button(screen1, text="Continue", command=continue_from_screen1)
    continue_button.pack(pady=20)

    # Start screen 1
    screen1.mainloop()

# Zone selection screen
def show_screen2():
    global screen2, zone_buttons, selected_zones_label


    screen2 = tk.Tk()
    screen2.title("Zone Selection")

    screen2.attributes('-topmost', True)
    screen2.update()
    
    instruction_label = tk.Label(screen2, text="Select the zones where the object is located:")
    instruction_label.pack(pady=10)


    main_frame = tk.Frame(screen2)
    main_frame.pack()

    # Create the robot's grid and allow the user
    # to click on the zones an object is in
    grid_frame = tk.Frame(main_frame)
    grid_frame.pack()
    columns = [chr(i) for i in range(ord('a'), ord('q'))] 
    rows = [str(i) for i in range(1, 10)] 

    # Display and format the grid
    zone_buttons = {}
    for row_index, row_label in enumerate(rows):
        for col_index, col_label in enumerate(columns):
            zone = f"{col_label}{row_label}"
            button = tk.Button(
                grid_frame,
                text=zone,
                width=5,
                command=lambda z=zone: toggle_zone(z)
            )
            button.grid(row=row_index, column=col_index, padx=2, pady=2)
            zone_buttons[zone] = button

    selected_zones_label = tk.Label(screen2, text="Selected Zones: None")
    selected_zones_label.pack(pady=10)

    continue_button = tk.Button(screen2, text="Continue", command=continue_from_screen2)
    continue_button.pack(pady=10)

    # Start screen 2
    screen2.mainloop()

# Data file selection screen
def show_screen3():
    global screen3, file_option_var, filename_entry, browse_button, selected_file_label


    screen3 = tk.Tk()
    screen3.title("File Selection")

    screen3.attributes('-topmost', True)
    screen3.update()
    instruction_label = tk.Label(screen3, text="Select the file to store the data:")
    instruction_label.pack(pady=10)

    # Allow the user to enter in the filename manually
    file_option_var = tk.StringVar(value="New File")
    new_file_option = tk.Radiobutton(screen3, text="New File", variable=file_option_var, value="New File")
    new_file_option.pack(anchor='w', padx=20)

    # Open a file explorer window to select an existing file
    existing_file_option = tk.Radiobutton(screen3, text="Existing File", variable=file_option_var, value="Existing File")
    existing_file_option.pack(anchor='w', padx=20)
    filename_entry = tk.Entry(screen3)
    filename_entry.pack(pady=5, padx=20, fill='x')
    browse_button = tk.Button(screen3, text="Browse...", command=browse_file)
    browse_button.pack(pady=5)

    selected_file_label = tk.Label(screen3, text="")
    selected_file_label.pack(pady=5)

    on_file_option_change()
    file_option_var.trace_add('write', on_file_option_change)

    continue_button = tk.Button(screen3, text="Continue", command=continue_from_screen3)
    continue_button.pack(pady=20)

    # Start screen 3
    screen3.mainloop()

# Specify how many rows of data want to be collected
def show_screen4():
    global screen4, rows_option_var, rows_entry


    screen4 = tk.Tk()
    screen4.title("Data Rows Selection")

    screen4.attributes('-topmost', True)
    screen4.update()
    instruction_label = tk.Label(screen4, text="How many rows of data do you want to collect?")
    instruction_label.pack(pady=10)
    
    # Default option of 500 rows
    rows_option_var = tk.StringVar(value="Default")
    default_option = tk.Radiobutton(screen4, text="Default (500 rows)", variable=rows_option_var, value="Default")
    default_option.pack(anchor='w', padx=20)

    # Allow the user to enter in a custom amount of rows
    custom_option = tk.Radiobutton(screen4, text="Custom", variable=rows_option_var, value="Custom")
    custom_option.pack(anchor='w', padx=20)

    rows_entry = tk.Entry(screen4)
    rows_entry.pack(pady=5, padx=20, fill='x')
    rows_entry.configure(state='disabled')

    on_rows_option_change()

    rows_option_var.trace_add('write', on_rows_option_change)
    continue_button = tk.Button(screen4, text="Continue", command=continue_from_screen4)
    continue_button.pack(pady=20)
    
    # Start screen 4
    screen4.mainloop()

def get_config_info():
    show_screen1()
    print(robot_name, selected_zones, filename, num_rows)
    return robot_name, selected_zones, filename, num_rows

