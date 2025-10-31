from .setdata import X_diab, y_diab
import pytest
import GA

def test_output():
    assert isinstance(GA.select(X_diab, y_diab), dict)

