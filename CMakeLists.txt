idf_component_register(SRCS "main.c" "on_board_ctl.c"
                    INCLUDE_DIRS "."
                    REQUIRES esp_driver_gpio esp_driver_uart esp_driver_twai esp_driver_pcnt freertos esp_timer driver)
