import matplotlib.pyplot as plt

def sort_and_plot(list1, list2):
    # Sort the lists in descending order
    sorted_list1 = sorted(list1, reverse=True)
    sorted_list2 = sorted(list2, reverse=True)

    # Import matplotlib for plotting


    # Plotting the sorted lists
    plt.figure(figsize=(10, 5))

    # Plotting both lists with their index on the x-axis and their value on the y-axis
    plt.plot(sorted_list1, 'o-', label='List 1')
    plt.plot(sorted_list2, 's-', label='List 2')

    # Adding titles and labels
    plt.title('Sorted Lists Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Showing the legend
    plt.legend()

    # Display the plot
    plt.show()


