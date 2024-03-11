from app.services.translate import service_translate
from app.businesses.fujinet_translate.bus import FujiTranslateBusiness

bus_translate = FujiTranslateBusiness(
    services_class=service_translate
)
