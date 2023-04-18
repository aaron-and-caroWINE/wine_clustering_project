import scipy.stats as  stats
import pandas as pd
from wrangle_wine import wrangle_wine

train, validate, test,\
train_scaled, validate_scaled, test_scaled = wrangle_wine()

train['is_high_quality'] = (train.quality == 7) | (train.quality == 6)

## VOLATILE ACIDITY STATS FUNCTIONS
def get_va_lv():
    '''
    Actions: gets stats test for volaitile acidity
    Imports:
        import scipy.stats as stats
        import pandas as pd
        from wrangle_wine import wrangle_wine
    '''
    
    # set alpha
    α = 0.05
    
    # get the levene stat and p-value
    lev, p = stats.levene(train.volatile_acidity[train.is_high_quality == 1], train.volatile_acidity[train.is_high_quality == 0])

    # if p-value is less than alpha
    if p < α:

        # print statement and p-value
        print(f'We can reject the null hypothesis with a p-value of {p}')

    else:

        # print statement and p-value
        print(f'We CANNOT reject the null hypothesis with a p-value of {p}')

    return


def get_va_mw():
    '''
    Actions: gets stats test for volaitile acidity
    Imports:
        from scipy.stats import stats
        import pandas as pd
        from wrangle_wine import wrangle_wine
    '''
    # set alpha
    α = 0.05
    
    # runn stats test 
    mw, p = stats.mannwhitneyu(train.volatile_acidity[train.is_high_quality == 1], train.volatile_acidity[train.is_high_quality == 0])

    # if p-value is less than alpha        
    if p < α:

        # print statement and p-value
        print(f'We can reject the null hypothesis with a p-value of {p}')

    else:

        # print statement and p-value
        print(f'We CANNOT reject the null hypothesis with a p-value of {p}')

    # exit function
    return 

## CHLORIDES  stats test
def get_chlorides_lv():
    '''
    Actions: gets stats test for chlorides
    Imports:
        from scipy.stats import stats
        import pandas as pd
        from wrangle_wine import wrangle_wine
    '''
    # set alpha
    α = 0.05
    
    # get the levene stat and p-value
    lev, p = stats.levene(train.chlorides[train.is_high_quality == 1], train.chlorides[train.is_high_quality == 0])

    # if p-value is less than alpha
    if p < α:

        # print statement and p-value
        print(f'We can reject the null hypothesis with a p-value of {p}')

    else:

        # print statement and p-value
        print(f'We CANNOT reject the null hypothesis with a p-value of {p}')

    return


def get_chlorides_mw():
    '''
    Actions: gets stats test for chlorides
    Imports:
        from scipy.stats import stats
        import pandas as pd
        from wrangle_wine import wrangle_wine
    '''
    # set alpha
    α = 0.05
    
    # runn stats test 
    mw, p = stats.mannwhitneyu(train.chlorides[train.is_high_quality == 1], train.chlorides[train.is_high_quality == 0])

    # if p-value is less than alpha        
    if p < α:

        # print statement and p-value
        print(f'We can reject the null hypothesis with a p-value of {p}')

    else:

        # print statement and p-value
        print(f'We CANNOT reject the null hypothesis with a p-value of {p}')

    # exit function
    return 


# DENSITY STATS TESTS
def get_density_lv():
    '''
    Actions: gets stats test for density
    Imports:
        from scipy.stats import stats
        import pandas as pd
        from wrangle_wine import wrangle_wine
    '''
    # set alpha
    α = 0.05
    
    # get the levene stat and p-value
    lev, p = stats.levene(train.density[train.is_high_quality == 1], train.density[train.is_high_quality == 0])

    # if p-value is less than alpha
    if p < α:

        # print statement and p-value
        print(f'We can reject the null hypothesis with a p-value of {p}')

    else:

        # print statement and p-value
        print(f'We CANNOT reject the null hypothesis with a p-value of {p}')

    return


def get_density_mw():
    '''
    Actions: gets stats test for density
    Imports:
        from scipy.stats import stats
        import pandas as pd
        from wrangle_wine import wrangle_wine
    '''
    # set alpha
    α = 0.05
    
    # runn stats test 
    mw, p = stats.mannwhitneyu(train.density[train.is_high_quality == 1], train.density[train.is_high_quality == 0])

    # if p-value is less than alpha        
    if p < α:

        # print statement and p-value
        print(f'We can reject the null hypothesis with a p-value of {p}')

    else:

        # print statement and p-value
        print(f'We CANNOT reject the null hypothesis with a p-value of {p}')

    # exit function
    return 


# ALCOHOL stats tests
def get_alcohol_lv():
    '''
    Actions: gets stats test for alcohol
    Imports:
        from scipy.stats import stats
        import pandas as pd
        from wrangle_wine import wrangle_wine
    '''
    # set alpha
    α = 0.05
    
    # get the levene stat and p-value
    lev, p = stats.levene(train.alcohol[train.is_high_quality == 1], train.alcohol[train.is_high_quality == 0])

    # if p-value is less than alpha
    if p < α:

        # print statement and p-value
        print(f'We can reject the null hypothesis with a p-value of {p}')

    else:

        # print statement and p-value
        print(f'We CANNOT reject the null hypothesis with a p-value of {p}')

    return


def get_alcohol_mw():
    '''
    Actions: gets stats test for alcohol
    Imports:
        from scipy.stats import stats
        import pandas as pd
        from wrangle_wine import wrangle_wine
    '''
    # set alpha
    α = 0.05
    
    # runn stats test 
    mw, p = stats.mannwhitneyu(train.alcohol[train.is_high_quality == 1], train.alcohol[train.is_high_quality == 0])

    # if p-value is less than alpha        
    if p < α:

        # print statement and p-value
        print(f'We can reject the null hypothesis with a p-value of {p}')

    else:

        # print statement and p-value
        print(f'We CANNOT reject the null hypothesis with a p-value of {p}')

    # exit function
    return 