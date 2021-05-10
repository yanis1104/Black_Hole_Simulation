##
## EPITECH PROJECT, 2020
## TEK2
## File description:
## Makefile
##

BINARY	=	simulation

CC	=	@gcc
CXX	=	@g++
RM	=	@rm -rf

SRC	=	$(wildcard src/*.cpp)
OBJ	=	$(SRC:.cpp=.o)

CXXFLAGS	=	-Wextra -Wall -Werror

CPPFLAGS	=	-I include/

LDFLAGS		=	# linker flags

# ALL #############################################################################################
all: $(OBJ)
	$(CXX) $(LDFLAGS) -o $(BINARY) $(SRC) $(CPPFLAGS) -lsfml-graphics -lsfml-window -lsfml-system
	@$(RM) $(OBJ)
	$(PRINT) $(OK) "Compilation finished" "\033[0m"

clean:
	$(RM) $(OBJ)
	$(PRINT) $(REMOVE) "OBJECT" $(NO_COLOR)
	$(RM) vgcore.*
	$(PRINT) $(REMOVE) "VALGRIND VGCORE" $(NO_COLOR)
	$(PRINT) $(OK) "Clean" "\033[0m"

clean_ut:
	$(RM) $(BINARY_UT)
	$(RM) $(OBJ_UT)
	$(RM) *.gcda
	$(RM) *.gcno

fclean: clean clean_ut
	$(RM) $(BINARY)
	$(PRINT) $(REMOVE) "Binary name: $(BINARY)" "\033[0m"

re er: fclean all

.PHONY: all clean clean_ut fclean run tests_run re er help

# COLOR ###########################################################################################
NOCOLOR	= 	"\033[0m"
RED		= 	"\033[0;31m"
CYAN	= 	"\033[0;32m"
BLUE	= 	"\033[0;34m"
VIOLET	= 	"\033[1;36m"
WHITE	= 	"\033[1;37m"

# SHORTCUT ########################################################################################
OK		=	 $(RED) "[OK]" $(BLUE)
REMOVE	=	$(CYAN) "[RM]" $(BLUE)
PRINT	=	@echo -e