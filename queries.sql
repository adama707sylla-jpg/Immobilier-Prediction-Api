select count(*)
from maisons;

select count(*) - count("GrLivArea") as manquantes
from maisons;