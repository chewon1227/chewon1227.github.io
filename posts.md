---
layout: default
title: Posts by Category
permalink: /posts
nav_order: 2
---

{% for category in site.categories %}
## {{ category[0] | capitalize }}

<ul>
  {% for post in category[1] %}
    <li><a href="{{ post.url }}">{{ post.title }}</a> ({{ post.date | date: "%Y-%m-%d" }})</li>
  {% endfor %}
</ul>

{% endfor %}