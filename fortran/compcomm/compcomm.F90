! drive computation and communication
! split comp and comm
! adjust duration of comp and comm
! repeat one pair of comp and comm


      module EXTRAE_MODULE
          interface
              subroutine extrae_user_function (enter)
                  integer*4, intent(in) :: enter
              end subroutine extrae_user_function
              subroutine extrae_next_hwc_set
              end subroutine extrae_next_hwc_set
!        subroutine extrae_init
!        end subroutine extrae_init
!        subroutine extrae_fini
!        end subroutine extrae_fini
          end interface
      end module EXTRAE_MODULE

      program compcomm

          integer, parameter :: NUMREPEAT = 1000, LENARRAY=10000
          real(kind=8), dimension(LENARRAY) :: array
          real, dimension(LENARRAY, 2) :: shuffle

          array = 0
          call random_seed()

          do iter=1, NUMREPEAT
              call extrae_user_function (1)
              call comp(array, shuffle)
              call comm(array, shuffle)
              call extrae_user_function (0)
              call extrae_next_hwc_set
          end do

          print *, "sum of array: ", sum(array)
      contains

      subroutine comp(array, shuffle)
          real(kind=8), intent(out), dimension(:) :: array
          real, intent(out), dimension(:, :) :: shuffle
          call random_number(array)
          call random_number(shuffle)
      end subroutine

      subroutine comm(array, shuffle)
          real(kind=8), intent(inout), dimension(:) :: array
          real, intent(in), dimension(:, :) :: shuffle
          real(kind=8) r
          integer n, k

          do n = lbound(array,1)+1, ubound(array, 1)
              k = int(shuffle(n, 1)*n) + 1
              r = array(n)
              array(n) = array(k)
              array(k) = r
          end do

      end subroutine

      end program
