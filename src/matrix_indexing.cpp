#include <iostream>

int main()
{
    const int rows = 3;
    const int cols = 4;

    float A[rows * cols] = {
        1, 2, 3, 4, 
        5, 6, 7, 8,
        9, 10, 11, 12
    };

    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            int index = row * cols + col;

            std::cout
                << "A[" << row << "][" << col << "]"
                << " -> index " << index
                << " -> value " << A[index]
                << '\n';
        }
    }

    return 0;
}
