def ehrlich_aberth(coefficients, x0, max_iterations): # Initialize the variables 
    x_n = x0;x_n_1 = x0;x_n_2 = x0;iteration = 0

    # Calculate the coefficients of the polynomial
    a0 = coefficients[0]
    a1 = coefficients[1]
    a2 = coefficients[2]
    a3 = coefficients[3]
    a4 = coefficients[4]
    a5 = coefficients[5]

    # Iterate until the maximum number of iterations is reached
    while iteration < max_iterations:
        # Calculate the next iteration
        x_n = x_n_1 - (a0 + a1*x_n_1 + a2*x_n_1**2 + a3*x_n_1**3 + a4*x_n_1**4 + a5*x_n_1**5) / (a1 + 2*a2*x_n_1 + 3*a3*x_n_1**2 + 4*a4*x_n_1**3 + 5*a5*x_n_1**4)
        
        # Check for convergence
        if abs(x_n - x_n_1) < 0.00001:
            break
        
        # Update the variables
        x_n_2 = x_n_1
        x_n_1 = x_n
        iteration += 1

    # Return the root
    return x_n
coff=[-1,0,0,0,0,1]
result=ehrlich_aberth(coff,-1,100)
print(result)