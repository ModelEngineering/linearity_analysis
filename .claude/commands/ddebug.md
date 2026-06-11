# Detailed debugging for $ARGUMENTS

This command sets up a test file for detailed debugging of a single test.

1. Verify that the IGNORE_TESTS guard is present and used in all tests.
2. Set IGNORE_TESTS to True.
3. If ARGUMENTS is the name of the test method to debug. 
   1. Comment out its guard.
   2. Set IGNORE_TESTS to True.
4. If ARUMENTS is the null string
   1. Remove all commented guards
   2. Change IGNORE_TESTS to False
5. Do not run any tests.
6. Edit files without permissions.