// lib/pages/start_page.dart
import 'package:flutter/material.dart';
import 'package:flutter_custom_clippers/flutter_custom_clippers.dart';
import 'package:indrive_ai/customer_page.dart';
import 'package:indrive_ai/driver/driver_form_page.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../theme/app_colors.dart';

class StartPage extends StatefulWidget {
  const StartPage({super.key});

  @override
  State<StartPage> createState() => _StartPageState();
}

class _StartPageState extends State<StartPage>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<Offset> _logoAnimation;
  late Animation<Offset> _driverButtonAnimation;
  late Animation<Offset> _customerButtonAnimation;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();

    //clearPrefs();
    _controller =
        AnimationController(vsync: this, duration: const Duration(seconds: 2));

    _logoAnimation = Tween<Offset>(
      begin: const Offset(0, -1),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeOut));

    _driverButtonAnimation = Tween<Offset>(
      begin: const Offset(-1, 0),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeOutBack));

    _customerButtonAnimation = Tween<Offset>(
      begin: const Offset(1, 0),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeOutBack));

    _fadeAnimation = CurvedAnimation(parent: _controller, curve: Curves.easeIn);

    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void clearPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.clear();
  }

  Widget _buildButton(
      {required String text,
      required VoidCallback onTap,
      required Color background,
      required Color textColor}) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 8),
      child: SizedBox(
        width: double.infinity,
        child: ElevatedButton(
          style: ElevatedButton.styleFrom(
            backgroundColor: background,
            padding: const EdgeInsets.symmetric(vertical: 16),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(40),
            ),
            elevation: 4,
            shadowColor: background,
          ),
          onPressed: onTap,
          child: Text(
            text,
            style: TextStyle(
                fontSize: 18, fontWeight: FontWeight.bold, color: textColor),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: Stack(
        children: [
          // Верхняя волна
          ClipPath(
            clipper: WaveClipperOne(),
            child: Container(
              height: 250,
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  colors: [AppColors.primary, AppColors.surface],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
              ),
            ),
          ),
          // Нижняя волна
          Align(
            alignment: Alignment.bottomCenter,
            child: ClipPath(
              clipper: WaveClipperTwo(reverse: true),
              child: Container(
                height: 200,
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      AppColors.surface,
                      AppColors.primary.withOpacity(0.7),
                    ],
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                  ),
                ),
              ),
            ),
          ),

          // Контент
          SafeArea(
            child: FadeTransition(
              opacity: _fadeAnimation,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // Лого с анимацией выезда сверху
                  SlideTransition(
                      position: _logoAnimation,
                      child: Image.asset(
                        "assets/img/logo.png",
                        width: 180,
                      )),
                  const SizedBox(height: 80),

                  // Кнопка "Я водитель"
                  SlideTransition(
                    position: _driverButtonAnimation,
                    child: _buildButton(
                      text: "Я водитель",
                      background: AppColors.primary,
                      textColor: AppColors.textPrimary,
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (_) => const DriverFormPage()),
                        );
                      },
                    ),
                  ),

                  // Кнопка "Я заказчик"
                  SlideTransition(
                    position: _customerButtonAnimation,
                    child: _buildButton(
                      text: "Я заказчик",
                      background: AppColors.surface,
                      textColor: AppColors.textPrimary,
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (_) => const CustomerPage()),
                        );
                      },
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
