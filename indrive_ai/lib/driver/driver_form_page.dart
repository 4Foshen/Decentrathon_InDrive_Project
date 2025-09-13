import 'package:flutter/material.dart';
import 'car_photo_page.dart';
import 'driver_storage.dart';
import 'package:indrive_ai/theme/app_colors.dart';

class DriverFormPage extends StatefulWidget {
  const DriverFormPage({super.key});

  @override
  State<DriverFormPage> createState() => _DriverFormPageState();
}

class _DriverFormPageState extends State<DriverFormPage> {
  final TextEditingController nameController = TextEditingController();
  final TextEditingController carController = TextEditingController();

  void _goNext() {
    if (nameController.text.isEmpty || carController.text.isEmpty) return;

    Navigator.push(
      context,
      buildPageTransition(
        CarPhotoPage(
          driverName: nameController.text,
          carModel: carController.text,
        ),
        type: TransitionType.slide,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text("Регистрация водителя"),
        backgroundColor: Colors.transparent,
        elevation: 0,
        foregroundColor: AppColors.textPrimary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Введите данные",
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: AppColors.textPrimary,
              ),
            ),
            const SizedBox(height: 24),

            // Поле имени
            TextField(
              controller: nameController,
              style: const TextStyle(color: AppColors.textPrimary),
              decoration: InputDecoration(
                labelText: "Имя водителя",
                labelStyle: const TextStyle(color: AppColors.textSecondary),
                filled: true,
                fillColor: AppColors.surface,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide.none,
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: AppColors.primary),
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Поле марки машины
            TextField(
              controller: carController,
              style: const TextStyle(color: AppColors.textPrimary),
              decoration: InputDecoration(
                labelText: "Марка машины",
                labelStyle: const TextStyle(color: AppColors.textSecondary),
                filled: true,
                fillColor: AppColors.surface,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide.none,
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: AppColors.primary),
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Кнопка
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _goNext,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.primary,
                  padding:
                      const EdgeInsets.symmetric(vertical: 14, horizontal: 30),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                  elevation: 3,
                ),
                child: const Text(
                  "Далее",
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textPrimary,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Универсальные транзишены
Route buildPageTransition(Widget page,
    {TransitionType type = TransitionType.slide}) {
  return PageRouteBuilder(
    transitionDuration: const Duration(milliseconds: 600),
    pageBuilder: (_, __, ___) => page,
    transitionsBuilder: (_, animation, __, child) {
      switch (type) {
        case TransitionType.slide:
          return SlideTransition(
            position: Tween(begin: const Offset(1, 0), end: Offset.zero)
                .animate(CurvedAnimation(
                    parent: animation, curve: Curves.easeOutCubic)),
            child: FadeTransition(opacity: animation, child: child),
          );
        case TransitionType.scale:
          return ScaleTransition(
            scale: Tween(begin: 0.9, end: 1.0).animate(
                CurvedAnimation(parent: animation, curve: Curves.easeOutBack)),
            child: FadeTransition(opacity: animation, child: child),
          );
        default:
          return child;
      }
    },
  );
}

enum TransitionType { slide, scale }
