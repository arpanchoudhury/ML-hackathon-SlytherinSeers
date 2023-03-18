
subroutine fortran_gaussian_kernel(A, B, A_col, B_col, K, sigma)

    implicit none
    
    double precision, dimension(:,:), intent(in) :: A, B
    integer, intent(in) :: A_col, B_col
    double precision, intent(in) :: sigma

    double precision, dimension(:,:), intent(inout) :: K

    double precision :: inv_sigma
    double precision, allocatable, dimension(:) :: tmp

    integer :: i, j

    inv_sigma = -0.5d0/(sigma**2)

    allocate(tmp(size(A, dim=1)))

    do i = 1, A_col
        do j = 1, B_col
            tmp(:) = A(:,i) - B(:,j)
            k(i,j) = exp(inv_sigma * sum(tmp**2))
        enddo
    enddo

    deallocate(tmp)

end subroutine fortran_gaussian_kernel



subroutine fortran_laplacian_kernel(A, B, A_col, B_col, K, sigma)

    implicit none

    double precision, dimension(:,:), intent(in) :: A, B
    integer, intent(in) :: A_col, B_col
    double precision, intent(in) :: sigma

    double precision, dimension(:,:), intent(inout) :: K

    double precision :: inv_sigma

    integer :: i, j

    inv_sigma = -1.0d0/sigma

    do i = 1, A_col
        do j = 1, B_col
            K(i,j) = exp(inv_sigma * sum(abs(A(:,i) - B(:,j))))
        enddo
    enddo

end subroutine fortran_laplacian_kernel

